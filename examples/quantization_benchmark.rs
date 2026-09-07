//! Quantization comparison benchmark
//!
//! Run with: cargo run --example quantization_benchmark --release
//!
//! Builds one SPFresh index per quantizer and outputs data for the
//! compression ratio vs recall chart.

use diskann_rs::{DistL2, PQConfig, QuantizerKind, SPFresh, SPFreshConfig};
use rand::prelude::*;
use rand::SeedableRng;
use std::collections::HashSet;
use std::time::Instant;

fn main() {
    let dim: usize = 128;
    let n_vectors = 10_000;
    let n_queries = 100;
    let k = 10;
    let n_probe = 8;

    println!("Quantization Benchmark (SPFresh)");
    println!("================================");
    println!(
        "Vectors: {}, Dim: {}, Queries: {}, k: {}, n_probe: {}\n",
        n_vectors, dim, n_queries, k, n_probe
    );

    // Generate random vectors
    let mut rng = StdRng::seed_from_u64(42);
    let vectors: Vec<Vec<f32>> = (0..n_vectors)
        .map(|_| (0..dim).map(|_| rng.r#gen::<f32>() * 2.0 - 1.0).collect())
        .collect();
    let queries: Vec<Vec<f32>> = (0..n_queries)
        .map(|_| (0..dim).map(|_| rng.r#gen::<f32>() * 2.0 - 1.0).collect())
        .collect();

    // Compute ground truth (exact k-NN)
    let ground_truth: Vec<Vec<u64>> = queries
        .iter()
        .map(|q| {
            let mut dists: Vec<(u64, f32)> = vectors
                .iter()
                .enumerate()
                .map(|(i, v)| (i as u64, l2_squared(q, v)))
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            dists.iter().take(k).map(|(i, _)| *i).collect()
        })
        .collect();

    let pq = |num_subspaces| PQConfig {
        num_subspaces,
        num_centroids: 256,
        kmeans_iterations: 15,
        training_sample_size: 5000,
    };

    // (method, quantizer, code size in bytes)
    let variants = [
        ("None", None, dim * 4),
        ("F16", Some(QuantizerKind::F16), dim * 2),
        ("Int8", Some(QuantizerKind::Int8), dim),
        ("PQ-32", Some(QuantizerKind::PQ(pq(32))), 32),
        ("PQ-16", Some(QuantizerKind::PQ(pq(16))), 16),
        ("PQ-8", Some(QuantizerKind::PQ(pq(8))), 8),
        (
            "RaBitQ",
            Some(QuantizerKind::RaBitQ),
            dim.max(64).next_power_of_two() / 8 + 8,
        ),
    ];

    println!(
        "| Method | Compression | Code Size | Build Time | Search Time | Recall@{} |",
        k
    );
    println!("|--------|-------------|-----------|------------|-------------|----------|");

    let mut rows = Vec::new();
    for (name, quantizer, code_size) in variants {
        let path = format!("bench_quantized_{}", rows.len());
        let cfg = SPFreshConfig {
            max_posting_size: 128,
            ..Default::default()
        };
        let compression = (dim * 4) as f32 / code_size as f32;

        let start = Instant::now();
        let index = SPFresh::<DistL2>::build(&vectors, &path, cfg, quantizer).unwrap();
        let build_time = start.elapsed();

        let start = Instant::now();
        let recall = compute_recall(&index, &queries, &ground_truth, k, n_probe);
        let search_time = start.elapsed();

        println!(
            "| {} | {:.1}x | {} B | {:.1}ms | {:.1}ms | {:.1}% |",
            name,
            compression,
            code_size,
            build_time.as_secs_f64() * 1000.0,
            search_time.as_secs_f64() * 1000.0,
            recall * 100.0
        );
        rows.push((name, compression, recall));
        cleanup(&path);
    }

    println!("\n# Chart Data (CSV)");
    println!("method,compression,recall");
    for (name, compression, recall) in rows {
        println!("{},{:.1},{:.1}", name, compression, recall * 100.0);
    }
}

fn cleanup(path: &str) {
    for suffix in ["spf", "postings", "vectors", "centroids", "centroids.base"] {
        let _ = std::fs::remove_file(format!("{path}.{suffix}"));
    }
}

fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum()
}

fn compute_recall(
    index: &SPFresh<DistL2>,
    queries: &[Vec<f32>],
    ground_truth: &[Vec<u64>],
    k: usize,
    n_probe: usize,
) -> f32 {
    let mut total_recall = 0.0;
    for (query, gt) in queries.iter().zip(ground_truth) {
        let retrieved: HashSet<u64> = index.search(query, k, n_probe).into_iter().collect();
        let gt_set: HashSet<u64> = gt.iter().copied().collect();
        let hits = retrieved.intersection(&gt_set).count();
        total_recall += hits as f32 / k as f32;
    }
    total_recall / queries.len() as f32
}
