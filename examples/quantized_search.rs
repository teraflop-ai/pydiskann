//! Quantized search example — demonstrates building and searching an SPFresh
//! index with F16, Int8, PQ, and RaBitQ quantized postings.
//!
//! Run with: cargo run --example quantized_search --release

use diskann_rs::{DistL2, PQConfig, QuantizerKind, SPFresh, SPFreshConfig};
use rand::prelude::*;
use rand::SeedableRng;
use std::collections::HashSet;
use std::time::Instant;

fn main() {
    let dim = 64;
    let n_vectors = 2_000;
    let n_queries = 50;
    let k = 10;
    let n_probe = 8;

    println!("Quantized SPFresh Search Example");
    println!("================================");
    println!(
        "Vectors: {}, Dim: {}, Queries: {}, k: {}, n_probe: {}\n",
        n_vectors, dim, n_queries, k, n_probe
    );

    // Generate random vectors
    let mut rng = StdRng::seed_from_u64(42);
    let vectors: Vec<Vec<f32>> = (0..n_vectors)
        .map(|_| (0..dim).map(|_| rng.r#gen::<f32>()).collect())
        .collect();
    let queries: Vec<Vec<f32>> = (0..n_queries)
        .map(|_| (0..dim).map(|_| rng.r#gen::<f32>()).collect())
        .collect();

    // Ground truth (brute force)
    let ground_truth: Vec<Vec<u64>> = queries
        .iter()
        .map(|q| {
            let mut dists: Vec<(u64, f32)> = vectors
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let d: f32 = q.iter().zip(v).map(|(a, b)| (a - b) * (a - b)).sum();
                    (i as u64, d)
                })
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            dists.iter().take(k).map(|(i, _)| *i).collect()
        })
        .collect();

    let pq_config = PQConfig {
        num_subspaces: 8,
        num_centroids: 256,
        kmeans_iterations: 15,
        training_sample_size: 0,
    };

    // (method, quantizer, rerank_size, rerank label)
    let variants = [
        ("F16", QuantizerKind::F16, 0, "none"),
        ("Int8", QuantizerKind::Int8, 0, "none"),
        ("PQ-8", QuantizerKind::PQ(pq_config), 0, "none"),
        ("PQ-8", QuantizerKind::PQ(pq_config), 50, "top-50"),
        ("RaBitQ", QuantizerKind::RaBitQ, 0, "bound"),
    ];

    println!(
        "| {:<12} | {:<10} | {:<12} | {:<12} | {:<10} |",
        "Method", "Rerank", "Build (ms)", "Search (ms)", "Recall@10"
    );
    println!(
        "|{:-<14}|{:-<12}|{:-<14}|{:-<14}|{:-<12}|",
        "", "", "", "", ""
    );

    for (i, (name, quantizer, rerank_size, rerank)) in variants.into_iter().enumerate() {
        let path = format!("example_quantized_{i}");
        let cfg = SPFreshConfig {
            max_posting_size: 64,
            rerank_size,
            ..Default::default()
        };

        let start = Instant::now();
        let index = SPFresh::<DistL2>::build(&vectors, &path, cfg, Some(quantizer)).unwrap();
        let build_ms = start.elapsed().as_secs_f64() * 1000.0;

        let start = Instant::now();
        let recall = avg_recall(&index, &queries, &ground_truth, k, n_probe);
        let search_ms = start.elapsed().as_secs_f64() * 1000.0;

        println!(
            "| {:<12} | {:<10} | {:<12.1} | {:<12.1} | {:>9.1}% |",
            name,
            rerank,
            build_ms,
            search_ms,
            recall * 100.0
        );
        cleanup(&path);
    }

    println!("\nDone.");
}

fn cleanup(path: &str) {
    for suffix in ["spf", "postings", "vectors", "centroids", "centroids.base"] {
        let _ = std::fs::remove_file(format!("{path}.{suffix}"));
    }
}

fn avg_recall(
    index: &SPFresh<DistL2>,
    queries: &[Vec<f32>],
    ground_truth: &[Vec<u64>],
    k: usize,
    n_probe: usize,
) -> f32 {
    let mut total = 0.0f32;
    for (query, gt) in queries.iter().zip(ground_truth) {
        let results = index.search(query, k, n_probe);
        let gt_set: HashSet<u64> = gt.iter().copied().collect();
        let hits = results.iter().filter(|id| gt_set.contains(id)).count();
        total += hits as f32 / k as f32;
    }
    total / queries.len() as f32
}
