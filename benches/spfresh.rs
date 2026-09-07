//! SPFresh benchmark: build, insert throughput per batch (catches degradation),
//! search QPS / recall@10 vs n_probe, delete + gc, quantizer variants, and a
//! plain DiskANN reference.
//!
//! Run: cargo bench --bench spfresh
//! Env: SPF_N (default 100000), SPF_DIM (128), SPF_Q (500), SPF_ONLY (raw|f16|int8|rabitq|diskann)

use diskann_rs::{DiskANN, DistL2, QuantizerKind, SPFresh, SPFreshConfig};
use rand::prelude::*;
use rayon::prelude::*;
use std::collections::HashSet;
use std::time::Instant;

const K: usize = 10;
const PROBES: [usize; 6] = [1, 2, 4, 8, 16, 32];

fn env(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn clustered(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let centers: Vec<Vec<f32>> = (0..200)
        .map(|_| (0..dim).map(|_| rng.r#gen::<f32>() * 2.0 - 1.0).collect())
        .collect();
    (0..n)
        .map(|i| {
            let c = &centers[i % centers.len()];
            c.iter().map(|x| x + (rng.r#gen::<f32>() - 0.5) * 0.4).collect()
        })
        .collect()
}

fn ground_truth(queries: &[Vec<f32>], data: &[Vec<f32>], live: &HashSet<u64>) -> Vec<Vec<u64>> {
    queries
        .par_iter()
        .map(|q| {
            let mut d: Vec<(u64, f32)> = data
                .iter()
                .enumerate()
                .filter(|(i, _)| live.contains(&(*i as u64)))
                .map(|(i, v)| (i as u64, q.iter().zip(v).map(|(a, b)| (a - b) * (a - b)).sum()))
                .collect();
            d.sort_by(|a, b| a.1.total_cmp(&b.1));
            d.iter().take(K).map(|(i, _)| *i).collect()
        })
        .collect()
}

fn recall(res: &[Vec<u64>], gt: &[Vec<u64>]) -> f64 {
    let hits: usize = res
        .iter()
        .zip(gt)
        .map(|(r, g)| {
            let g: HashSet<u64> = g.iter().copied().collect();
            r.iter().filter(|i| g.contains(i)).count()
        })
        .sum();
    hits as f64 / (K * gt.len()) as f64
}

fn search_row(tag: &str, idx: &SPFresh<DistL2>, queries: &[Vec<f32>], gt: &[Vec<u64>]) {
    print!("{:<22}", tag);
    for &p in &PROBES {
        let t = Instant::now();
        let res: Vec<Vec<u64>> = queries.iter().map(|q| idx.search(q, K, p)).collect();
        let qps = queries.len() as f64 / t.elapsed().as_secs_f64();
        print!(" | {:>6.0} qps {:>5.1}%", qps, recall(&res, gt) * 100.0);
    }
    println!();
}

fn run_variant(name: &str, quantizer: Option<QuantizerKind>, rerank: usize, data: &[Vec<f32>], queries: &[Vec<f32>], n_build: usize) {
    let path = format!("bench_spf_{}", name);
    let cfg = SPFreshConfig { rerank_size: rerank, ..Default::default() };
    println!("\n== {} (rerank={}) ==", name, rerank);
    println!("{:<22}{}", "n_probe", PROBES.iter().map(|p| format!(" | {:>15}", p)).collect::<String>());

    let t = Instant::now();
    let idx = SPFresh::<DistL2>::build(&data[..n_build], &path, cfg, quantizer).unwrap();
    let build_s = t.elapsed().as_secs_f64();
    let mut live: HashSet<u64> = (0..n_build as u64).collect();
    let mut gt = ground_truth(queries, data, &live);
    println!("build {} vecs: {:.2}s ({:.0} vec/s) {:?}", n_build, build_s, n_build as f64 / build_s, idx.stats());
    search_row("after build", &idx, queries, &gt);

    let batch = (data.len() - n_build) / 4;
    for b in 0..4 {
        let lo = n_build + b * batch;
        let t = Instant::now();
        let ids = idx.insert(&data[lo..lo + batch]).unwrap();
        assert_eq!(ids[0], lo as u64);
        let s = t.elapsed().as_secs_f64();
        live.extend(ids);
        println!("insert batch {} ({} vecs): {:.2}s ({:.0} vec/s) max_posting={}", b, batch, s, batch as f64 / s, idx.stats().max_posting);
    }
    gt = ground_truth(queries, data, &live);
    search_row("after inserts", &idx, queries, &gt);

    let dead: Vec<u64> = (0..live.len() as u64).step_by(10).collect();
    let t = Instant::now();
    idx.delete(&dead);
    let del_s = t.elapsed().as_secs_f64();
    for d in &dead {
        live.remove(d);
    }
    let t = Instant::now();
    idx.gc().unwrap();
    let gc_s = t.elapsed().as_secs_f64();
    gt = ground_truth(queries, data, &live);
    println!("delete {}: {:.3}s, gc: {:.2}s {:?}", dead.len(), del_s, gc_s, idx.stats());
    search_row("after delete+gc", &idx, queries, &gt);

    let t = Instant::now();
    idx.compact().unwrap();
    println!("compact: {:.2}s", t.elapsed().as_secs_f64());
    search_row("after compact", &idx, queries, &gt);

    let t = Instant::now();
    idx.save().unwrap();
    println!("save: {:.2}s", t.elapsed().as_secs_f64());
    for suffix in ["spf", "postings", "vectors", "centroids", "centroids.base", "cache"] {
        let _ = std::fs::remove_file(format!("{}.{}", path, suffix));
    }
}

fn diskann_reference(data: &[Vec<f32>], queries: &[Vec<f32>]) {
    let live: HashSet<u64> = (0..data.len() as u64).collect();
    let gt = ground_truth(queries, data, &live);
    let path = "bench_spf_reference.db";
    let t = Instant::now();
    let idx = DiskANN::<DistL2>::build_index_default(data, DistL2, path).unwrap();
    println!("\n== DiskANN reference: build {} vecs {:.2}s ==", data.len(), t.elapsed().as_secs_f64());
    for beam in [32, 64, 128, 256] {
        let t = Instant::now();
        let res: Vec<Vec<u64>> = queries
            .iter()
            .map(|q| idx.search(q, K, beam).into_iter().map(u64::from).collect())
            .collect();
        let qps = queries.len() as f64 / t.elapsed().as_secs_f64();
        println!("beam {:>3}: {:>6.0} qps {:>5.1}%", beam, qps, recall(&res, &gt) * 100.0);
    }
    let _ = std::fs::remove_file(path);
}

fn main() {
    let (n, dim, q) = (env("SPF_N", 100_000), env("SPF_DIM", 128), env("SPF_Q", 500));
    println!("SPFresh bench: n={} dim={} queries={} k={} threads={}", n, dim, q, K, rayon::current_num_threads());
    let data = clustered(n, dim, 1);
    let queries = clustered(q, dim, 2);
    let n_build = n * 6 / 10;

    let only = std::env::var("SPF_ONLY").ok();
    let want = |name: &str| only.as_deref().map_or(true, |o| o == name);
    if want("raw") {
        run_variant("raw", None, 0, &data, &queries, n_build);
    }
    if want("f16") {
        run_variant("f16", Some(QuantizerKind::F16), 0, &data, &queries, n_build);
    }
    if want("int8") {
        run_variant("int8_rerank", Some(QuantizerKind::Int8), 50, &data, &queries, n_build);
    }
    if want("rabitq") {
        run_variant("rabitq_rerank", Some(QuantizerKind::RaBitQ), 100, &data, &queries, n_build);
    }
    if want("diskann") {
        diskann_reference(&data, &queries);
    }
}
