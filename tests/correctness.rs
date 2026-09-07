//! Comprehensive correctness and robustness tests for diskann-rs.
//!
//! These integration tests exercise the public API with focus on:
//! - Search invariants (sorted, no duplicates, correct count)
//! - Edge cases (single vector, k > n, beam_width=1)
//! - Graph connectivity (all nodes reachable from medoid)
//! - Error handling (corrupted files, invalid configs)
//! - Recall regression on deterministic datasets
//! - Cross-module consistency

use diskann_rs::pq::{PQConfig, ProductQuantizer};
use diskann_rs::sq::{F16Quantizer, Int8Quantizer, VectorQuantizer};
use diskann_rs::{
    DiskANN, DiskAnnParams, DistCosine, DistL2, Filter, FilteredDiskANN, IncrementalConfig,
    IncrementalDiskANN, IncrementalQuantizedConfig, QuantizedConfig, QuantizedDiskANN,
    QuantizerKind,
};
use rand::prelude::*;
use rand::SeedableRng;
use std::collections::HashSet;

// =========================================================================
// Helpers
// =========================================================================

fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| (0..dim).map(|_| rng.r#gen::<f32>()).collect())
        .collect()
}

fn brute_force_knn_l2(vectors: &[Vec<f32>], query: &[f32], k: usize) -> Vec<u32> {
    let mut dists: Vec<(u32, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let d: f32 = query
                .iter()
                .zip(v)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
                .sqrt();
            (i as u32, d)
        })
        .collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    dists.iter().take(k).map(|(i, _)| *i).collect()
}

fn recall_at_k(retrieved: &[u32], ground_truth: &[u32]) -> f32 {
    let gt_set: HashSet<u32> = ground_truth.iter().copied().collect();
    let hits = retrieved.iter().filter(|id| gt_set.contains(id)).count();
    hits as f32 / ground_truth.len().max(1) as f32
}

/// Assert search result invariants: no duplicates, sorted distances, correct count.
fn assert_search_invariants(results: &[(u32, f32)], k: usize, num_vectors: usize) {
    let expected_count = k.min(num_vectors);
    assert_eq!(
        results.len(),
        expected_count,
        "Expected {} results, got {}",
        expected_count,
        results.len()
    );

    // No duplicate IDs
    let ids: HashSet<u32> = results.iter().map(|(id, _)| *id).collect();
    assert_eq!(
        ids.len(),
        results.len(),
        "Duplicate IDs in search results: {:?}",
        results
    );

    // Distances are non-negative
    for (id, dist) in results {
        assert!(*dist >= 0.0, "Negative distance for id {}: {}", id, dist);
        assert!(
            dist.is_finite(),
            "Non-finite distance for id {}: {}",
            id,
            dist
        );
    }

    // Distances are non-decreasing
    for pair in results.windows(2) {
        assert!(
            pair[0].1 <= pair[1].1 + 1e-6,
            "Results not sorted: ({}, {}) > ({}, {})",
            pair[0].0,
            pair[0].1,
            pair[1].0,
            pair[1].1,
        );
    }
}

fn default_ann_params() -> DiskAnnParams {
    DiskAnnParams {
        max_degree: 32,
        build_beam_width: 128,
        alpha: 1.2,
    }
}

fn cleanup(paths: &[&str]) {
    for p in paths {
        let _ = std::fs::remove_file(p);
    }
}

// =========================================================================
// 1. Search Invariant Tests
// =========================================================================

#[test]
fn test_search_invariants_base_index() {
    let path = "test_invariants_base.db";
    cleanup(&[path]);

    let vectors = random_vectors(200, 32, 100);
    let index =
        DiskANN::<DistL2>::build_index_with_params(&vectors, DistL2 {}, path, default_ann_params())
            .unwrap();

    for k in [1, 5, 10, 50] {
        let query = &vectors[0];
        let results = index.search_with_dists(query, k, 64);
        assert_search_invariants(&results, k, 200);
    }

    cleanup(&[path]);
}

#[test]
fn test_search_invariants_quantized_pq() {
    let path = "test_invariants_quant_pq.db";
    cleanup(&[path]);

    let vectors = random_vectors(200, 32, 101);
    let pq_config = PQConfig {
        num_subspaces: 4,
        num_centroids: 64,
        kmeans_iterations: 10,
        training_sample_size: 0,
    };
    let index = QuantizedDiskANN::<DistL2>::build_pq(
        &vectors,
        DistL2 {},
        path,
        default_ann_params(),
        pq_config,
        QuantizedConfig { rerank_size: 0 },
    )
    .unwrap();

    for k in [1, 5, 10, 50] {
        let results = index.search_with_dists(&vectors[0], k, 64);
        assert_search_invariants(&results, k, 200);
    }

    cleanup(&[path]);
}

#[test]
fn test_search_invariants_quantized_f16() {
    let path = "test_invariants_quant_f16.db";
    cleanup(&[path]);

    let vectors = random_vectors(200, 32, 102);
    let index = QuantizedDiskANN::<DistL2>::build_f16(
        &vectors,
        DistL2 {},
        path,
        default_ann_params(),
        QuantizedConfig { rerank_size: 0 },
    )
    .unwrap();

    for k in [1, 5, 10, 50] {
        let results = index.search_with_dists(&vectors[0], k, 64);
        assert_search_invariants(&results, k, 200);
    }

    cleanup(&[path]);
}

#[test]
fn test_search_invariants_quantized_int8() {
    let path = "test_invariants_quant_int8.db";
    cleanup(&[path]);

    let vectors = random_vectors(200, 32, 103);
    let index = QuantizedDiskANN::<DistL2>::build_int8(
        &vectors,
        DistL2 {},
        path,
        default_ann_params(),
        QuantizedConfig { rerank_size: 0 },
    )
    .unwrap();

    for k in [1, 5, 10, 50] {
        let results = index.search_with_dists(&vectors[0], k, 64);
        assert_search_invariants(&results, k, 200);
    }

    cleanup(&[path]);
}

#[test]
fn test_search_invariants_quantized_reranked() {
    let path = "test_invariants_quant_rerank.db";
    cleanup(&[path]);

    let vectors = random_vectors(200, 32, 104);
    let pq_config = PQConfig {
        num_subspaces: 4,
        num_centroids: 64,
        kmeans_iterations: 10,
        training_sample_size: 0,
    };
    let index = QuantizedDiskANN::<DistL2>::build_pq(
        &vectors,
        DistL2 {},
        path,
        default_ann_params(),
        pq_config,
        QuantizedConfig { rerank_size: 50 },
    )
    .unwrap();

    for k in [1, 5, 10] {
        let results = index.search_with_dists(&vectors[0], k, 64);
        assert_search_invariants(&results, k, 200);
    }

    cleanup(&[path]);
}

#[test]
fn test_search_invariants_filtered() {
    let path = "test_invariants_filtered";
    cleanup(&[&format!("{}.idx", path), &format!("{}.labels", path)]);

    let vectors = random_vectors(200, 32, 105);
    let labels: Vec<Vec<u64>> = (0..200).map(|i| vec![i as u64 % 5]).collect();
    let index = FilteredDiskANN::<DistL2>::build(&vectors, &labels, path).unwrap();

    let query = &vectors[0];
    let results = index.search_filtered(query, 10, 64, &Filter::None);
    assert_eq!(results.len(), 10);
    let ids: HashSet<u32> = results.iter().copied().collect();
    assert_eq!(
        ids.len(),
        results.len(),
        "Duplicate IDs in filtered results"
    );

    cleanup(&[&format!("{}.idx", path), &format!("{}.labels", path)]);
}

#[test]
fn test_search_invariants_incremental() {
    let path = "test_invariants_incr.db";
    cleanup(&[path]);

    let vectors = random_vectors(100, 32, 106);
    let index = IncrementalDiskANN::<DistL2>::build_default(&vectors, path).unwrap();

    let query = &vectors[0];
    let results = index.search_with_dists(query, 10, 64);
    assert_eq!(results.len(), 10);

    // No duplicates
    let ids: HashSet<u64> = results.iter().map(|(id, _)| *id).collect();
    assert_eq!(ids.len(), results.len());

    // Sorted
    for pair in results.windows(2) {
        assert!(pair[0].1 <= pair[1].1 + 1e-6);
    }

    cleanup(&[path]);
}

// =========================================================================
// 2. Edge Case Tests
// =========================================================================

#[test]
fn test_single_vector_index() {
    let path = "test_single_vec.db";
    cleanup(&[path]);

    let vectors = vec![vec![1.0, 2.0, 3.0]];
    let index =
        DiskANN::<DistL2>::build_index_with_params(&vectors, DistL2 {}, path, default_ann_params())
            .unwrap();

    assert_eq!(index.num_vectors, 1);
    assert_eq!(index.dim, 3);

    // Search for 1 neighbor — should return the only vector
    let results = index.search(&[1.0, 2.0, 3.0], 1, 8);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0], 0);

    // Search for k > n — should return only 1 result
    let results = index.search(&[0.0, 0.0, 0.0], 10, 8);
    assert_eq!(results.len(), 1);

    cleanup(&[path]);
}

#[test]
fn test_two_vector_index() {
    let path = "test_two_vec.db";
    cleanup(&[path]);

    let vectors = vec![vec![0.0, 0.0], vec![10.0, 10.0]];
    let index =
        DiskANN::<DistL2>::build_index_with_params(&vectors, DistL2 {}, path, default_ann_params())
            .unwrap();

    // Query close to first vector
    let results = index.search(&[0.1, 0.1], 2, 8);
    assert_eq!(results.len(), 2);
    assert_eq!(results[0], 0);
    assert_eq!(results[1], 1);

    // Query close to second vector
    let results = index.search(&[9.9, 9.9], 2, 8);
    assert_eq!(results.len(), 2);
    assert_eq!(results[0], 1);
    assert_eq!(results[1], 0);

    cleanup(&[path]);
}

#[test]
fn test_k_greater_than_num_vectors() {
    let path = "test_k_gt_n.db";
    cleanup(&[path]);

    let vectors = random_vectors(5, 16, 200);
    let index =
        DiskANN::<DistL2>::build_index_with_params(&vectors, DistL2 {}, path, default_ann_params())
            .unwrap();

    let results = index.search_with_dists(&vectors[0], 100, 64);
    assert_eq!(results.len(), 5, "Should return all 5 vectors when k=100");
    assert_search_invariants(&results, 100, 5);

    cleanup(&[path]);
}

#[test]
fn test_beam_width_one() {
    let path = "test_beam1.db";
    cleanup(&[path]);

    let vectors = random_vectors(50, 16, 201);
    let index =
        DiskANN::<DistL2>::build_index_with_params(&vectors, DistL2 {}, path, default_ann_params())
            .unwrap();

    // beam_width=1 is a degenerate case — should still return valid results
    let results = index.search_with_dists(&vectors[0], 1, 1);
    assert_eq!(results.len(), 1);
    assert!(results[0].1 >= 0.0);

    cleanup(&[path]);
}

#[test]
fn test_identical_vectors() {
    let path = "test_identical.db";
    cleanup(&[path]);

    // All vectors are the same
    let vectors = vec![vec![1.0f32, 2.0, 3.0]; 10];
    let index =
        DiskANN::<DistL2>::build_index_with_params(&vectors, DistL2 {}, path, default_ann_params())
            .unwrap();

    let results = index.search_with_dists(&[1.0, 2.0, 3.0], 5, 8);
    assert_eq!(results.len(), 5);
    // All distances should be 0
    for (_, dist) in &results {
        assert!(
            *dist < 1e-6,
            "Distance to identical vector should be ~0, got {}",
            dist
        );
    }

    cleanup(&[path]);
}

#[test]
fn test_empty_vectors_rejected() {
    let path = "test_empty.db";
    cleanup(&[path]);

    let vectors: Vec<Vec<f32>> = vec![];
    let result = DiskANN::<DistL2>::build_index_default(&vectors, DistL2 {}, path);
    assert!(result.is_err(), "Empty vectors should be rejected");

    cleanup(&[path]);
}

#[test]
fn test_dimension_mismatch_rejected() {
    let path = "test_dim_mismatch.db";
    cleanup(&[path]);

    let vectors = vec![vec![1.0, 2.0], vec![1.0, 2.0, 3.0]];
    let result = DiskANN::<DistL2>::build_index_default(&vectors, DistL2 {}, path);
    assert!(result.is_err(), "Mismatched dimensions should be rejected");

    cleanup(&[path]);
}

#[test]
fn test_quantized_single_vector() {
    let path = "test_quant_single.db";
    cleanup(&[path]);

    let vectors = vec![vec![1.0f32; 32]];
    let index = QuantizedDiskANN::<DistL2>::build_f16(
        &vectors,
        DistL2 {},
        path,
        default_ann_params(),
        QuantizedConfig { rerank_size: 0 },
    )
    .unwrap();

    let results = index.search(&[1.0f32; 32], 5, 8);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0], 0);

    cleanup(&[path]);
}

#[test]
fn test_quantized_k_greater_than_n() {
    let path = "test_quant_k_gt_n.db";
    cleanup(&[path]);

    let vectors = random_vectors(5, 32, 202);
    let index = QuantizedDiskANN::<DistL2>::build_int8(
        &vectors,
        DistL2 {},
        path,
        default_ann_params(),
        QuantizedConfig { rerank_size: 0 },
    )
    .unwrap();

    let results = index.search_with_dists(&vectors[0], 100, 64);
    assert_eq!(results.len(), 5);
    assert_search_invariants(&results, 100, 5);

    cleanup(&[path]);
}

// =========================================================================
// 3. PQ Validation Tests
// =========================================================================

#[test]
fn test_pq_num_centroids_exceeds_256() {
    let vectors = random_vectors(100, 64, 300);
    let config = PQConfig {
        num_subspaces: 8,
        num_centroids: 512,
        kmeans_iterations: 5,
        training_sample_size: 0,
    };
    let result = ProductQuantizer::train(&vectors, config);
    assert!(result.is_err(), "num_centroids > 256 should be rejected");
}

#[test]
fn test_pq_dim_not_divisible_by_subspaces() {
    let vectors = random_vectors(100, 65, 301);
    let config = PQConfig {
        num_subspaces: 8,
        num_centroids: 256,
        kmeans_iterations: 5,
        training_sample_size: 0,
    };
    let result = ProductQuantizer::train(&vectors, config);
    assert!(result.is_err(), "Dimension 65 not divisible by 8 subspaces");
}

#[test]
fn test_pq_single_subspace() {
    let vectors = random_vectors(200, 32, 302);
    let config = PQConfig {
        num_subspaces: 1,
        num_centroids: 256,
        kmeans_iterations: 10,
        training_sample_size: 0,
    };
    let pq = ProductQuantizer::train(&vectors, config).unwrap();
    let stats = pq.stats();
    assert_eq!(stats.num_subspaces, 1);
    assert_eq!(stats.code_size_bytes, 1);

    // Should still produce valid codes
    let codes = pq.encode(&vectors[0]);
    assert_eq!(codes.len(), 1);
    let decoded = pq.decode(&codes);
    assert_eq!(decoded.len(), 32);
}

#[test]
fn test_pq_max_subspaces() {
    // num_subspaces == dim: each byte represents one dimension
    let vectors = random_vectors(200, 32, 303);
    let config = PQConfig {
        num_subspaces: 32,
        num_centroids: 256,
        kmeans_iterations: 10,
        training_sample_size: 0,
    };
    let pq = ProductQuantizer::train(&vectors, config).unwrap();
    let codes = pq.encode(&vectors[0]);
    assert_eq!(codes.len(), 32);

    // Recall should be high with 32 subspaces
    let query = &vectors[0];
    let table = pq.create_distance_table(query);
    let mut dists: Vec<(usize, f32)> = vectors
        .iter()
        .enumerate()
        .skip(1)
        .map(|(i, v)| (i, pq.distance_with_table(&table, &pq.encode(v))))
        .collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let pq_top10: HashSet<usize> = dists.iter().take(10).map(|(i, _)| *i).collect();
    let gt = brute_force_knn_l2(&vectors, query, 10);
    let gt_set: HashSet<usize> = gt.iter().map(|&i| i as usize).collect();
    let recall = pq_top10.intersection(&gt_set).count() as f32 / 10.0;
    assert!(
        recall >= 0.6,
        "PQ-32 recall@10 should be >= 0.6, got {}",
        recall
    );
}

// =========================================================================
// 4. Graph Connectivity Tests
// =========================================================================

#[test]
fn test_graph_connectivity_from_medoid() {
    let path = "test_connectivity.db";
    cleanup(&[path]);

    let vectors = random_vectors(500, 16, 400);
    let index = DiskANN::<DistL2>::build_index_with_params(
        &vectors,
        DistL2 {},
        path,
        DiskAnnParams {
            max_degree: 16,
            build_beam_width: 64,
            alpha: 1.2,
        },
    )
    .unwrap();

    // Check that search can find every vector when queried with itself
    let found_count = (0..500)
        .filter(|&i| {
            let q = &vectors[i];
            let results = index.search(q, 1, 64);
            results.contains(&(i as u32))
        })
        .count();

    let reachability = found_count as f32 / 500.0;
    assert!(
        reachability > 0.95,
        "Reachability from medoid should be > 95%, got {:.1}% ({}/500)",
        reachability * 100.0,
        found_count
    );

    cleanup(&[path]);
}

#[test]
fn test_graph_connectivity_small() {
    let path = "test_connectivity_small.db";
    cleanup(&[path]);

    // Small graph — every node should be findable
    let vectors = random_vectors(20, 8, 401);
    let index = DiskANN::<DistL2>::build_index_with_params(
        &vectors,
        DistL2 {},
        path,
        DiskAnnParams {
            max_degree: 8,
            build_beam_width: 32,
            alpha: 1.5,
        },
    )
    .unwrap();

    // With high beam width, we should find every single vector
    for i in 0..20 {
        let q = &vectors[i];
        let results = index.search(q, 5, 32);
        assert!(
            results.contains(&(i as u32)),
            "Vector {} not found in its own search (results: {:?})",
            i,
            results
        );
    }

    cleanup(&[path]);
}

// =========================================================================
// 5. Corrupted Data Tests
// =========================================================================

#[test]
fn test_corrupted_index_file() {
    // Write garbage to a file and try to open it
    let path = "test_corrupted.db";
    cleanup(&[path]);

    std::fs::write(path, b"garbage data that is not an index").unwrap();
    let result = DiskANN::<DistL2>::open_index_with(path, DistL2 {});
    assert!(result.is_err(), "Should fail on corrupted index file");

    cleanup(&[path]);
}

#[test]
fn test_truncated_index_file() {
    let path = "test_truncated.db";
    cleanup(&[path]);

    // Write just 4 bytes (not enough for metadata length)
    std::fs::write(path, &[0u8; 4]).unwrap();
    let result = DiskANN::<DistL2>::open_index_with(path, DistL2 {});
    assert!(result.is_err(), "Should fail on truncated index file");

    cleanup(&[path]);
}

#[test]
fn test_corrupted_sidecar_file() {
    let base_path = "test_corrupt_sidecar_base.db";
    let sidecar_path = "test_corrupt_sidecar.qann";
    cleanup(&[base_path, sidecar_path]);

    let vectors = random_vectors(50, 32, 500);
    let _base = DiskANN::<DistL2>::build_index_with_params(
        &vectors,
        DistL2 {},
        base_path,
        default_ann_params(),
    )
    .unwrap();

    // Write garbage sidecar
    std::fs::write(sidecar_path, b"not a valid sidecar file").unwrap();

    let result = QuantizedDiskANN::<DistL2>::open(
        base_path,
        sidecar_path,
        DistL2 {},
        QuantizedConfig::default(),
    );
    assert!(result.is_err(), "Should fail on corrupted sidecar");

    cleanup(&[base_path, sidecar_path]);
}

#[test]
fn test_corrupted_bytes_too_small() {
    let result = DiskANN::<DistL2>::from_bytes(vec![0u8; 3], DistL2 {});
    assert!(result.is_err(), "3 bytes should be too small for metadata");
}

#[test]
fn test_quantized_from_bytes_too_small() {
    let result =
        QuantizedDiskANN::<DistL2>::from_bytes(&[0u8; 5], DistL2 {}, QuantizedConfig::default());
    assert!(result.is_err(), "5 bytes should be too small");
}

// =========================================================================
// 6. Recall Regression Tests (Deterministic)
// =========================================================================

#[test]
fn test_recall_regression_l2_1000v() {
    let path = "test_recall_l2_1000.db";
    cleanup(&[path]);

    let dim = 64;
    let n = 1000;
    let vectors = random_vectors(n, dim, 600);
    let index = DiskANN::<DistL2>::build_index_with_params(
        &vectors,
        DistL2 {},
        path,
        DiskAnnParams {
            max_degree: 32,
            build_beam_width: 128,
            alpha: 1.2,
        },
    )
    .unwrap();

    // Test recall over 50 queries
    let queries = random_vectors(50, dim, 601);
    let mut total_recall = 0.0f32;
    for query in &queries {
        let results = index.search(query, 10, 64);
        let gt = brute_force_knn_l2(&vectors, query, 10);
        total_recall += recall_at_k(&results, &gt);
    }
    let avg_recall = total_recall / 50.0;
    assert!(
        avg_recall >= 0.85,
        "L2 recall@10 on 1000 vectors should be >= 85%, got {:.1}%",
        avg_recall * 100.0
    );

    cleanup(&[path]);
}

#[test]
fn test_recall_regression_cosine_1000v() {
    let path = "test_recall_cos_1000.db";
    cleanup(&[path]);

    let dim = 64;
    let n = 1000;
    let vectors = random_vectors(n, dim, 602);
    let index = DiskANN::<DistCosine>::build_index_with_params(
        &vectors,
        DistCosine {},
        path,
        DiskAnnParams {
            max_degree: 32,
            build_beam_width: 128,
            alpha: 1.2,
        },
    )
    .unwrap();

    // For cosine, the search should still return reasonable results
    let queries = random_vectors(50, dim, 603);
    let mut all_found = true;
    for query in &queries {
        let results = index.search(query, 10, 64);
        if results.len() != 10 {
            all_found = false;
        }
    }
    assert!(all_found, "All queries should return exactly 10 results");

    cleanup(&[path]);
}

#[test]
fn test_recall_quantized_f16_vs_exact() {
    let path_exact = "test_recall_f16_exact.db";
    let path_quant = "test_recall_f16_quant.db";
    cleanup(&[path_exact, path_quant]);

    let dim = 64;
    let vectors = random_vectors(500, dim, 604);
    let ann_params = default_ann_params();

    let exact_index =
        DiskANN::<DistL2>::build_index_with_params(&vectors, DistL2 {}, path_exact, ann_params)
            .unwrap();

    let quant_index = QuantizedDiskANN::<DistL2>::build_f16(
        &vectors,
        DistL2 {},
        path_quant,
        ann_params,
        QuantizedConfig { rerank_size: 0 },
    )
    .unwrap();

    // F16 quantized search should have recall very close to exact graph search
    let queries = random_vectors(30, dim, 605);
    let mut agreement = 0.0f32;
    for query in &queries {
        let exact_res = exact_index.search(query, 10, 64);
        let quant_res = quant_index.search(query, 10, 64);
        let exact_set: HashSet<u32> = exact_res.iter().copied().collect();
        let quant_set: HashSet<u32> = quant_res.iter().copied().collect();
        agreement += exact_set.intersection(&quant_set).count() as f32 / 10.0;
    }
    let avg_agreement = agreement / 30.0;
    assert!(
        avg_agreement >= 0.80,
        "F16 quantized vs exact agreement should be >= 80%, got {:.1}%",
        avg_agreement * 100.0
    );

    cleanup(&[path_exact, path_quant]);
}

#[test]
fn test_recall_quantized_int8_vs_exact() {
    let path_exact = "test_recall_int8_exact.db";
    let path_quant = "test_recall_int8_quant.db";
    cleanup(&[path_exact, path_quant]);

    let dim = 64;
    let vectors = random_vectors(500, dim, 606);
    let ann_params = default_ann_params();

    let exact_index =
        DiskANN::<DistL2>::build_index_with_params(&vectors, DistL2 {}, path_exact, ann_params)
            .unwrap();

    let quant_index = QuantizedDiskANN::<DistL2>::build_int8(
        &vectors,
        DistL2 {},
        path_quant,
        ann_params,
        QuantizedConfig { rerank_size: 0 },
    )
    .unwrap();

    let queries = random_vectors(30, dim, 607);
    let mut agreement = 0.0f32;
    for query in &queries {
        let exact_res = exact_index.search(query, 10, 64);
        let quant_res = quant_index.search(query, 10, 64);
        let exact_set: HashSet<u32> = exact_res.iter().copied().collect();
        let quant_set: HashSet<u32> = quant_res.iter().copied().collect();
        agreement += exact_set.intersection(&quant_set).count() as f32 / 10.0;
    }
    let avg_agreement = agreement / 30.0;
    assert!(
        avg_agreement >= 0.75,
        "Int8 quantized vs exact agreement should be >= 75%, got {:.1}%",
        avg_agreement * 100.0
    );

    cleanup(&[path_exact, path_quant]);
}

#[test]
fn test_recall_reranking_strictly_helps() {
    let path_no = "test_recall_rr_no.db";
    let path_yes = "test_recall_rr_yes.db";
    cleanup(&[path_no, path_yes]);

    let dim = 64;
    let vectors = random_vectors(500, dim, 608);
    let pq_config = PQConfig {
        num_subspaces: 8,
        num_centroids: 256,
        kmeans_iterations: 15,
        training_sample_size: 0,
    };
    let ann_params = default_ann_params();

    let index_no = QuantizedDiskANN::<DistL2>::build_pq(
        &vectors,
        DistL2 {},
        path_no,
        ann_params,
        pq_config,
        QuantizedConfig { rerank_size: 0 },
    )
    .unwrap();

    let index_yes = QuantizedDiskANN::<DistL2>::build_pq(
        &vectors,
        DistL2 {},
        path_yes,
        ann_params,
        pq_config,
        QuantizedConfig { rerank_size: 100 },
    )
    .unwrap();

    let queries = random_vectors(30, dim, 609);
    let mut total_no = 0.0f32;
    let mut total_yes = 0.0f32;
    for query in &queries {
        let gt = brute_force_knn_l2(&vectors, query, 10);
        total_no += recall_at_k(&index_no.search(query, 10, 64), &gt);
        total_yes += recall_at_k(&index_yes.search(query, 10, 64), &gt);
    }
    let avg_no = total_no / 30.0;
    let avg_yes = total_yes / 30.0;
    assert!(
        avg_yes >= avg_no,
        "Reranking should improve recall: without={:.1}%, with={:.1}%",
        avg_no * 100.0,
        avg_yes * 100.0
    );

    cleanup(&[path_no, path_yes]);
}

// =========================================================================
// 7. Filtered Search Edge Cases
// =========================================================================

#[test]
fn test_filtered_no_matches() {
    let path = "test_filtered_no_match";
    cleanup(&[&format!("{}.idx", path), &format!("{}.labels", path)]);

    let vectors = random_vectors(100, 16, 700);
    let labels: Vec<Vec<u64>> = (0..100).map(|_| vec![0]).collect();
    let index = FilteredDiskANN::<DistL2>::build(&vectors, &labels, path).unwrap();

    // Filter for label == 999 — no matches
    let results = index.search_filtered(&vectors[0], 10, 64, &Filter::label_eq(0, 999));
    assert_eq!(results.len(), 0, "No vectors match label 999");

    cleanup(&[&format!("{}.idx", path), &format!("{}.labels", path)]);
}

#[test]
fn test_filtered_all_match() {
    let path = "test_filtered_all_match";
    cleanup(&[&format!("{}.idx", path), &format!("{}.labels", path)]);

    let vectors = random_vectors(100, 16, 701);
    let labels: Vec<Vec<u64>> = (0..100).map(|_| vec![1]).collect();
    let index = FilteredDiskANN::<DistL2>::build(&vectors, &labels, path).unwrap();

    // Filter for label == 1 — all match, should behave like unfiltered
    let filtered = index.search_filtered(&vectors[0], 10, 64, &Filter::label_eq(0, 1));
    let unfiltered = index.search_filtered(&vectors[0], 10, 64, &Filter::None);

    assert_eq!(filtered.len(), 10);
    assert_eq!(unfiltered.len(), 10);

    // Results should overlap significantly
    let f_set: HashSet<u32> = filtered.iter().copied().collect();
    let u_set: HashSet<u32> = unfiltered.iter().copied().collect();
    let overlap = f_set.intersection(&u_set).count();
    assert!(
        overlap >= 7,
        "All-match filter should approximate unfiltered: overlap {}/10",
        overlap
    );

    cleanup(&[&format!("{}.idx", path), &format!("{}.labels", path)]);
}

// =========================================================================
// 8. Incremental Edge Cases
// =========================================================================

#[test]
fn test_incremental_delete_all() {
    let path = "test_incr_del_all.db";
    cleanup(&[path]);

    let vectors = random_vectors(20, 16, 800);
    let index = IncrementalDiskANN::<DistL2>::build_default(&vectors, path).unwrap();

    // Delete all base vectors
    let ids: Vec<u64> = (0..20).collect();
    index.delete_vectors(&ids).unwrap();

    let results = index.search(&vectors[0], 10, 32);
    assert_eq!(results.len(), 0, "All deleted — should return empty");

    cleanup(&[path]);
}

#[test]
fn test_incremental_add_then_search() {
    let path = "test_incr_add_search.db";
    cleanup(&[path]);

    let vectors = random_vectors(50, 16, 801);
    let index = IncrementalDiskANN::<DistL2>::build_default(&vectors[..20], path).unwrap();

    // Add more vectors
    let new_ids = index.add_vectors(&vectors[20..30]).unwrap();
    assert_eq!(new_ids.len(), 10);

    let stats = index.stats();
    assert_eq!(stats.base_vectors, 20);
    assert_eq!(stats.delta_vectors, 10);
    assert_eq!(stats.total_live, 30);

    // Search should find results from both base and delta
    let results = index.search(&vectors[25], 5, 32);
    assert!(!results.is_empty());

    cleanup(&[path]);
}

#[test]
fn test_incremental_delete_idempotent() {
    let path = "test_incr_del_idemp.db";
    cleanup(&[path]);

    let vectors = random_vectors(20, 16, 802);
    let index = IncrementalDiskANN::<DistL2>::build_default(&vectors, path).unwrap();

    // Delete same ID twice — should not panic or error
    index.delete_vectors(&[0]).unwrap();
    index.delete_vectors(&[0]).unwrap();

    assert!(index.is_deleted(0));
    assert_eq!(index.stats().tombstones, 1);

    cleanup(&[path]);
}

// =========================================================================
// 9. Cross-Module Consistency
// =========================================================================

#[test]
fn test_quantized_batch_search_matches_individual() {
    let path = "test_batch_vs_single.db";
    cleanup(&[path]);

    let vectors = random_vectors(200, 32, 900);
    let index = QuantizedDiskANN::<DistL2>::build_f16(
        &vectors,
        DistL2 {},
        path,
        default_ann_params(),
        QuantizedConfig { rerank_size: 0 },
    )
    .unwrap();

    let queries: Vec<Vec<f32>> = vectors[0..10].to_vec();
    let batch_results = index.search_batch(&queries, 10, 64);

    for (i, query) in queries.iter().enumerate() {
        let single_results = index.search(query, 10, 64);
        assert_eq!(
            batch_results[i], single_results,
            "Batch result {} differs from single search",
            i
        );
    }

    cleanup(&[path]);
}

#[test]
fn test_quantized_persistence_preserves_codes() {
    let base_path = "test_codes_persist_base.db";
    let sidecar_path = "test_codes_persist.qann";
    cleanup(&[base_path, sidecar_path]);

    let vectors = random_vectors(100, 32, 901);
    let config = QuantizedConfig { rerank_size: 0 };

    let index = QuantizedDiskANN::<DistL2>::build_int8(
        &vectors,
        DistL2 {},
        base_path,
        default_ann_params(),
        config,
    )
    .unwrap();

    // Multiple queries before save
    let queries = random_vectors(10, 32, 902);
    let results_before: Vec<Vec<u32>> = queries.iter().map(|q| index.search(q, 5, 32)).collect();

    // Save and reload
    index.save_quantized(sidecar_path).unwrap();
    let loaded =
        QuantizedDiskANN::<DistL2>::open(base_path, sidecar_path, DistL2 {}, config).unwrap();

    // Same queries after reload
    let results_after: Vec<Vec<u32>> = queries.iter().map(|q| loaded.search(q, 5, 32)).collect();

    assert_eq!(
        results_before, results_after,
        "Results changed after save/load"
    );

    cleanup(&[base_path, sidecar_path]);
}

#[test]
fn test_to_bytes_from_bytes_preserves_all_quantizers() {
    for (label, seed) in [("pq", 910u64), ("f16", 911), ("int8", 912)] {
        let path = format!("test_bytes_rt_{}.db", label);
        cleanup(&[&path]);

        let vectors = random_vectors(100, 32, seed);
        let config = QuantizedConfig { rerank_size: 0 };
        let ann_params = default_ann_params();

        let index: QuantizedDiskANN<DistL2> = match label {
            "pq" => {
                let pq_config = PQConfig {
                    num_subspaces: 4,
                    num_centroids: 64,
                    kmeans_iterations: 10,
                    training_sample_size: 0,
                };
                QuantizedDiskANN::build_pq(
                    &vectors,
                    DistL2 {},
                    &path,
                    ann_params,
                    pq_config,
                    config,
                )
                .unwrap()
            }
            "f16" => {
                QuantizedDiskANN::build_f16(&vectors, DistL2 {}, &path, ann_params, config).unwrap()
            }
            "int8" => QuantizedDiskANN::build_int8(&vectors, DistL2 {}, &path, ann_params, config)
                .unwrap(),
            _ => unreachable!(),
        };

        let query = &vectors[0];
        let res_before = index.search(query, 5, 32);

        let bytes = index.to_bytes();
        let loaded = QuantizedDiskANN::<DistL2>::from_bytes(&bytes, DistL2 {}, config).unwrap();
        let res_after = loaded.search(query, 5, 32);

        assert_eq!(
            res_before, res_after,
            "to_bytes/from_bytes round-trip failed for {}",
            label
        );

        cleanup(&[&path]);
    }
}

// =========================================================================
// 10. Scalar Quantizer Edge Cases
// =========================================================================

#[test]
fn test_f16_extreme_values() {
    let q = F16Quantizer::new(4);
    // Test with values near f16 range limits
    let vector = vec![65504.0, -65504.0, 0.0, 1e-7]; // f16 max, min, zero, tiny
    let codes = q.encode(&vector);
    let decoded = q.decode(&codes);
    assert_eq!(decoded.len(), 4);
    // Large values should survive
    assert!((decoded[0] - 65504.0).abs() < 1.0);
    assert!((decoded[1] + 65504.0).abs() < 1.0);
    assert!((decoded[2] - 0.0).abs() < 1e-4);
}

#[test]
fn test_int8_single_vector_training() {
    // Training on a single vector: scale should be 1.0 (constant), not panic
    let vectors = vec![vec![1.0, 2.0, 3.0]];
    let q = Int8Quantizer::train(&vectors).unwrap();
    let codes = q.encode(&vectors[0]);
    let decoded = q.decode(&codes);
    // With a single vector, all dims are constant → scale=1.0, so decode is approximate
    assert_eq!(decoded.len(), 3);
}

#[test]
fn test_int8_negative_values() {
    let vectors = vec![
        vec![-10.0, -5.0, 0.0, 5.0, 10.0],
        vec![-8.0, -3.0, 2.0, 7.0, 12.0],
        vec![-12.0, -7.0, -2.0, 3.0, 8.0],
    ];
    let q = Int8Quantizer::train(&vectors).unwrap();

    for v in &vectors {
        let codes = q.encode(v);
        let decoded = q.decode(&codes);
        for (orig, dec) in v.iter().zip(&decoded) {
            assert!((orig - dec).abs() < 0.2, "orig={}, dec={}", orig, dec);
        }
    }
}

// =========================================================================
// 11. Beam Search vs Brute Force on Larger Dataset
// =========================================================================

#[test]
fn test_beam_search_matches_brute_force() {
    let path = "test_beam_vs_bf.db";
    cleanup(&[path]);

    let dim = 32;
    let n = 500;
    let vectors = random_vectors(n, dim, 1100);

    let index = DiskANN::<DistL2>::build_index_with_params(
        &vectors,
        DistL2 {},
        path,
        DiskAnnParams {
            max_degree: 32,
            build_beam_width: 128,
            alpha: 1.2,
        },
    )
    .unwrap();

    // Test with multiple queries using high beam width for best recall
    let queries = random_vectors(20, dim, 1101);
    let mut total_recall = 0.0f32;

    for query in &queries {
        let beam_results = index.search(query, 10, 128);
        let brute_results = brute_force_knn_l2(&vectors, query, 10);
        total_recall += recall_at_k(&beam_results, &brute_results);
    }

    let avg_recall = total_recall / 20.0;
    assert!(
        avg_recall >= 0.90,
        "Beam search recall@10 vs brute force should be >= 90%, got {:.1}%",
        avg_recall * 100.0
    );

    cleanup(&[path]);
}

// =========================================================================
// 12. Magic Number / Version Tests
// =========================================================================

#[test]
fn test_new_format_round_trip() {
    let path = "test_new_format_rt.db";
    cleanup(&[path]);

    let vectors = random_vectors(50, 16, 1200);
    let index =
        DiskANN::<DistL2>::build_index_with_params(&vectors, DistL2 {}, path, default_ann_params())
            .unwrap();

    let query = &vectors[0];
    let res_before = index.search(query, 5, 32);

    // Reopen from file
    let reopened = DiskANN::<DistL2>::open_index_with(path, DistL2 {}).unwrap();
    let res_after = reopened.search(query, 5, 32);
    assert_eq!(res_before, res_after);

    // Round-trip via bytes
    let bytes = index.to_bytes();
    let from_bytes = DiskANN::<DistL2>::from_bytes(bytes, DistL2 {}).unwrap();
    let res_bytes = from_bytes.search(query, 5, 32);
    assert_eq!(res_before, res_bytes);

    cleanup(&[path]);
}

// =========================================================================
// 13. Filtered + Quantized Search Tests
// =========================================================================

#[test]
fn test_quantized_filtered_search_basic() {
    let path = "test_quant_filtered_basic.db";
    cleanup(&[path]);

    let dim = 32;
    let n = 200;
    let vectors = random_vectors(n, dim, 1300);

    // Labels: category = i % 5
    let labels: Vec<Vec<u64>> = (0..n).map(|i| vec![i as u64 % 5]).collect();

    let index = QuantizedDiskANN::<DistL2>::build_f16(
        &vectors,
        DistL2 {},
        path,
        default_ann_params(),
        QuantizedConfig { rerank_size: 0 },
    )
    .unwrap();

    // Search for category 2
    let filter = Filter::label_eq(0, 2);
    let results = index.search_filtered(&vectors[0], 10, 64, &labels, &filter);

    // All results should be category 2
    for &id in &results {
        assert_eq!(
            labels[id as usize][0], 2,
            "Expected category 2 for id {}, got {}",
            id, labels[id as usize][0]
        );
    }

    // Should find at least some results
    assert!(
        !results.is_empty(),
        "Should find at least one matching result"
    );

    cleanup(&[path]);
}

#[test]
fn test_quantized_filtered_search_with_reranking() {
    let path = "test_quant_filtered_rerank.db";
    cleanup(&[path]);

    let dim = 32;
    let n = 200;
    let vectors = random_vectors(n, dim, 1301);
    let labels: Vec<Vec<u64>> = (0..n).map(|i| vec![i as u64 % 3]).collect();

    let index = QuantizedDiskANN::<DistL2>::build_f16(
        &vectors,
        DistL2 {},
        path,
        default_ann_params(),
        QuantizedConfig { rerank_size: 50 },
    )
    .unwrap();

    let filter = Filter::label_eq(0, 1);
    let results = index.search_filtered_with_dists(&vectors[0], 10, 64, &labels, &filter);

    // All results should be category 1
    for &(id, _) in &results {
        assert_eq!(labels[id as usize][0], 1);
    }

    // With reranking, distances should be exact
    for &(id, dist) in &results {
        let v = index.get_vector(id as usize);
        let exact: f32 = vectors[0]
            .iter()
            .zip(&v)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            .sqrt();
        assert!(
            (dist - exact).abs() < 1e-4,
            "Distance mismatch for id {}: returned {}, exact {}",
            id,
            dist,
            exact
        );
    }

    // Distances should be non-decreasing
    for pair in results.windows(2) {
        assert!(pair[0].1 <= pair[1].1 + 1e-6);
    }

    cleanup(&[path]);
}

#[test]
fn test_quantized_filtered_no_matches() {
    let path = "test_quant_filtered_none.db";
    cleanup(&[path]);

    let dim = 32;
    let n = 100;
    let vectors = random_vectors(n, dim, 1302);
    let labels: Vec<Vec<u64>> = (0..n).map(|_| vec![0]).collect();

    let index = QuantizedDiskANN::<DistL2>::build_f16(
        &vectors,
        DistL2 {},
        path,
        default_ann_params(),
        QuantizedConfig { rerank_size: 0 },
    )
    .unwrap();

    // Filter for non-existent label
    let results = index.search_filtered(&vectors[0], 10, 64, &labels, &Filter::label_eq(0, 999));
    assert_eq!(results.len(), 0, "No vectors match label 999");

    cleanup(&[path]);
}

#[test]
fn test_quantized_filtered_none_filter() {
    let path = "test_quant_filtered_none_filter.db";
    cleanup(&[path]);

    let dim = 32;
    let n = 100;
    let vectors = random_vectors(n, dim, 1303);
    let labels: Vec<Vec<u64>> = (0..n).map(|i| vec![i as u64 % 5]).collect();

    let index = QuantizedDiskANN::<DistL2>::build_f16(
        &vectors,
        DistL2 {},
        path,
        default_ann_params(),
        QuantizedConfig { rerank_size: 0 },
    )
    .unwrap();

    // Filter::None should behave like unfiltered search
    let filtered = index.search_filtered(&vectors[0], 10, 64, &labels, &Filter::None);
    let unfiltered = index.search(&vectors[0], 10, 64);
    assert_eq!(filtered, unfiltered);

    cleanup(&[path]);
}

// =========================================================================
// 14. Incremental + Filtered Search Tests
// =========================================================================

#[test]
fn test_incremental_filtered_basic() {
    let path = "test_incr_filt_basic.db";
    cleanup(&[path]);

    let vectors = random_vectors(100, 32, 1400);
    let labels: Vec<Vec<u64>> = (0..100).map(|i| vec![i as u64 % 5]).collect();

    let index = IncrementalDiskANN::<DistL2>::build_with_labels(
        &vectors,
        &labels,
        path,
        IncrementalConfig::default(),
    )
    .unwrap();

    assert!(index.has_labels());

    // Search with filter: only category 2
    let filter = Filter::label_eq(0, 2);
    let results = index.search_filtered(&vectors[0], 10, 64, &filter);

    // All results should be category 2
    for &id in &results {
        let idx = id as usize;
        assert_eq!(labels[idx][0], 2, "Expected category 2 for id {}", id);
    }

    assert!(!results.is_empty(), "Should find at least one result");

    cleanup(&[path]);
}

#[test]
fn test_incremental_filtered_add_with_labels() {
    let path = "test_incr_filt_add.db";
    cleanup(&[path]);

    let vectors = random_vectors(50, 32, 1401);
    let labels: Vec<Vec<u64>> = (0..50).map(|i| vec![i as u64 % 3]).collect();

    let index = IncrementalDiskANN::<DistL2>::build_with_labels(
        &vectors,
        &labels,
        path,
        IncrementalConfig::default(),
    )
    .unwrap();

    // Add vectors with label category 9 (unique, easy to test)
    let new_vecs = random_vectors(10, 32, 1402);
    let new_labels: Vec<Vec<u64>> = (0..10).map(|_| vec![9]).collect();
    let new_ids = index
        .add_vectors_with_labels(&new_vecs, &new_labels)
        .unwrap();
    assert_eq!(new_ids.len(), 10);

    // Search for category 9 — should find the delta vectors
    let filter = Filter::label_eq(0, 9);
    let results = index.search_filtered(&new_vecs[0], 5, 64, &filter);
    assert!(!results.is_empty(), "Should find added labeled vectors");

    cleanup(&[path]);
}

#[test]
fn test_incremental_filtered_delete() {
    let path = "test_incr_filt_del.db";
    cleanup(&[path]);

    let vectors = random_vectors(50, 32, 1403);
    let labels: Vec<Vec<u64>> = (0..50).map(|_| vec![1]).collect(); // all category 1

    let index = IncrementalDiskANN::<DistL2>::build_with_labels(
        &vectors,
        &labels,
        path,
        IncrementalConfig::default(),
    )
    .unwrap();

    // Delete some vectors
    index.delete_vectors(&[0, 1, 2]).unwrap();

    // Filtered search should exclude tombstoned vectors
    let filter = Filter::label_eq(0, 1);
    let results = index.search_filtered(&vectors[5], 10, 64, &filter);

    let result_set: HashSet<u64> = results.iter().copied().collect();
    assert!(!result_set.contains(&0), "Deleted id 0 should not appear");
    assert!(!result_set.contains(&1), "Deleted id 1 should not appear");
    assert!(!result_set.contains(&2), "Deleted id 2 should not appear");

    cleanup(&[path]);
}

#[test]
fn test_incremental_filtered_compact() {
    let path1 = "test_incr_filt_cmp1.db";
    let path2 = "test_incr_filt_cmp2.db";
    cleanup(&[path1, path2]);

    let vectors = random_vectors(50, 32, 1404);
    let labels: Vec<Vec<u64>> = (0..50).map(|i| vec![i as u64 % 3]).collect();

    let mut index = IncrementalDiskANN::<DistL2>::build_with_labels(
        &vectors,
        &labels,
        path1,
        IncrementalConfig::default(),
    )
    .unwrap();

    // Add some labeled vectors
    let new_vecs = random_vectors(10, 32, 1405);
    let new_labels: Vec<Vec<u64>> = (0..10).map(|_| vec![7]).collect();
    index
        .add_vectors_with_labels(&new_vecs, &new_labels)
        .unwrap();

    // Delete some base vectors
    index.delete_vectors(&[0, 1]).unwrap();

    // Compact
    index.compact(path2).unwrap();

    assert!(index.has_labels());
    let stats = index.stats();
    assert_eq!(stats.tombstones, 0);
    assert_eq!(stats.delta_vectors, 0);

    // Labels should survive compaction — search for category 7
    let filter = Filter::label_eq(0, 7);
    let results = index.search_filtered(&new_vecs[0], 5, 64, &filter);
    assert!(
        !results.is_empty(),
        "Category 7 vectors should survive compaction"
    );

    cleanup(&[path1, path2]);
}

// =========================================================================
// 15. Incremental + Quantized Search Tests
// =========================================================================

#[test]
fn test_incremental_quantized_f16() {
    let path = "test_incr_quant_f16.db";
    cleanup(&[path]);

    let vectors = random_vectors(100, 32, 1500);

    let index = IncrementalDiskANN::<DistL2>::build_quantized_f16(
        &vectors,
        path,
        IncrementalConfig::default(),
        IncrementalQuantizedConfig { rerank_size: 0 },
    )
    .unwrap();

    assert!(index.has_quantizer());

    let results = index.search(&vectors[0], 10, 64);
    assert_eq!(results.len(), 10);

    // Should have reasonable results (close to query vector)
    let best = index.get_vector(results[0]).unwrap();
    let dist: f32 = vectors[0]
        .iter()
        .zip(&best)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32>()
        .sqrt();
    assert!(dist < 2.0, "Best result too far: {}", dist);

    cleanup(&[path]);
}

#[test]
fn test_incremental_quantized_pq() {
    let path = "test_incr_quant_pq.db";
    cleanup(&[path]);

    let vectors = random_vectors(200, 32, 1501);
    let pq_config = PQConfig {
        num_subspaces: 4,
        num_centroids: 64,
        kmeans_iterations: 10,
        training_sample_size: 0,
    };

    let index = IncrementalDiskANN::<DistL2>::build_quantized_pq(
        &vectors,
        path,
        IncrementalConfig::default(),
        pq_config,
        IncrementalQuantizedConfig { rerank_size: 0 },
    )
    .unwrap();

    let results = index.search(&vectors[0], 10, 64);
    assert_eq!(results.len(), 10);

    cleanup(&[path]);
}

#[test]
fn test_incremental_quantized_add() {
    let path = "test_incr_quant_add.db";
    cleanup(&[path]);

    let vectors = random_vectors(50, 32, 1502);

    let index = IncrementalDiskANN::<DistL2>::build_quantized_f16(
        &vectors,
        path,
        IncrementalConfig::default(),
        IncrementalQuantizedConfig { rerank_size: 0 },
    )
    .unwrap();

    // Add more vectors
    let new_vecs = random_vectors(10, 32, 1503);
    index.add_vectors(&new_vecs).unwrap();

    let stats = index.stats();
    assert_eq!(stats.base_vectors, 50);
    assert_eq!(stats.delta_vectors, 10);

    // Search should work and find results from both base and delta
    let results = index.search(&new_vecs[0], 5, 64);
    assert!(!results.is_empty());

    cleanup(&[path]);
}

#[test]
fn test_incremental_quantized_compact() {
    let path1 = "test_incr_quant_cmp1.db";
    let path2 = "test_incr_quant_cmp2.db";
    cleanup(&[path1, path2]);

    let vectors = random_vectors(50, 32, 1504);

    let mut index = IncrementalDiskANN::<DistL2>::build_quantized_f16(
        &vectors,
        path1,
        IncrementalConfig::default(),
        IncrementalQuantizedConfig { rerank_size: 0 },
    )
    .unwrap();

    // Add and delete
    index.add_vectors(&random_vectors(10, 32, 1505)).unwrap();
    index.delete_vectors(&[0, 1]).unwrap();

    // Compact
    index.compact(path2).unwrap();

    assert!(index.has_quantizer());
    let stats = index.stats();
    assert_eq!(stats.tombstones, 0);
    assert_eq!(stats.delta_vectors, 0);

    // Search should still work
    let results = index.search(&vectors[5], 5, 64);
    assert!(!results.is_empty());

    cleanup(&[path1, path2]);
}

// =========================================================================
// 16. Full Combo Tests (Incremental + Filtered + Quantized)
// =========================================================================

#[test]
fn test_incremental_full_combo() {
    let path = "test_incr_full_combo.db";
    cleanup(&[path]);

    let dim = 32;
    let n = 100;
    let vectors = random_vectors(n, dim, 1600);
    let labels: Vec<Vec<u64>> = (0..n).map(|i| vec![i as u64 % 5]).collect();
    let pq_config = PQConfig {
        num_subspaces: 4,
        num_centroids: 64,
        kmeans_iterations: 10,
        training_sample_size: 0,
    };

    let index = IncrementalDiskANN::<DistL2>::build_full(
        &vectors,
        &labels,
        path,
        IncrementalConfig::default(),
        QuantizerKind::PQ(pq_config),
        IncrementalQuantizedConfig { rerank_size: 0 },
    )
    .unwrap();

    assert!(index.has_labels());
    assert!(index.has_quantizer());

    // Add labeled vectors
    let new_vecs = random_vectors(10, dim, 1601);
    let new_labels: Vec<Vec<u64>> = (0..10).map(|_| vec![8]).collect();
    index
        .add_vectors_with_labels(&new_vecs, &new_labels)
        .unwrap();

    // Delete some base vectors
    index.delete_vectors(&[0, 1, 2]).unwrap();

    // Unfiltered search
    let results = index.search(&vectors[5], 10, 64);
    let result_set: HashSet<u64> = results.iter().copied().collect();
    assert!(!result_set.contains(&0));
    assert!(!result_set.contains(&1));
    assert!(!result_set.contains(&2));

    // Filtered search: category 8 (delta only)
    let filter = Filter::label_eq(0, 8);
    let filtered_results = index.search_filtered(&new_vecs[0], 5, 64, &filter);
    assert!(
        !filtered_results.is_empty(),
        "Should find category 8 delta vectors"
    );

    // Filtered search: category 3 (base only)
    let filter3 = Filter::label_eq(0, 3);
    let filtered_base = index.search_filtered(&vectors[3], 5, 64, &filter3);
    for &id in &filtered_base {
        // id should be a base vector with label category 3
        let idx = id as usize;
        assert_eq!(labels[idx][0], 3);
    }

    cleanup(&[path]);
}

// =========================================================================
// 17. Serialization Roundtrip Tests
// =========================================================================

#[test]
fn test_incremental_filtered_bytes_roundtrip() {
    let path = "test_incr_filt_bytes.db";
    cleanup(&[path]);

    let vectors = random_vectors(50, 32, 1700);
    let labels: Vec<Vec<u64>> = (0..50).map(|i| vec![i as u64 % 3]).collect();

    let index = IncrementalDiskANN::<DistL2>::build_with_labels(
        &vectors,
        &labels,
        path,
        IncrementalConfig::default(),
    )
    .unwrap();

    // Add some labeled delta vectors
    let new_vecs = random_vectors(5, 32, 1701);
    let new_labels: Vec<Vec<u64>> = (0..5).map(|_| vec![9]).collect();
    index
        .add_vectors_with_labels(&new_vecs, &new_labels)
        .unwrap();
    index.delete_vectors(&[0]).unwrap();

    let bytes = index.to_bytes();

    let loaded =
        IncrementalDiskANN::<DistL2>::from_bytes(&bytes, DistL2 {}, IncrementalConfig::default())
            .unwrap();

    assert!(loaded.has_labels());
    let stats = loaded.stats();
    assert_eq!(stats.base_vectors, 50);
    assert_eq!(stats.delta_vectors, 5);
    assert_eq!(stats.tombstones, 1);

    // Filtered search should still work after roundtrip
    let filter = Filter::label_eq(0, 9);
    let results = loaded.search_filtered(&new_vecs[0], 3, 64, &filter);
    assert!(
        !results.is_empty(),
        "Should find category 9 after roundtrip"
    );

    cleanup(&[path]);
}

#[test]
fn test_incremental_quantized_bytes_roundtrip() {
    let path = "test_incr_quant_bytes.db";
    cleanup(&[path]);

    let vectors = random_vectors(50, 32, 1702);

    let index = IncrementalDiskANN::<DistL2>::build_quantized_f16(
        &vectors,
        path,
        IncrementalConfig::default(),
        IncrementalQuantizedConfig { rerank_size: 0 },
    )
    .unwrap();

    index.add_vectors(&random_vectors(5, 32, 1703)).unwrap();

    let query = &vectors[0];
    let res_before = index.search(query, 5, 64);

    let bytes = index.to_bytes();
    let loaded =
        IncrementalDiskANN::<DistL2>::from_bytes(&bytes, DistL2 {}, IncrementalConfig::default())
            .unwrap();

    assert!(loaded.has_quantizer());
    let res_after = loaded.search(query, 5, 64);

    // Results should be the same after roundtrip
    assert_eq!(
        res_before, res_after,
        "Results changed after bytes roundtrip"
    );

    cleanup(&[path]);
}

#[test]
fn test_incremental_backward_compat_bytes() {
    let path = "test_incr_compat_integ.db";
    cleanup(&[path]);

    let vectors = random_vectors(30, 16, 1704);

    // Build old-format bytes: [has_base:u8][base_len:u64][base_bytes][dim:u64]...
    let base =
        DiskANN::<DistL2>::build_index_with_params(&vectors, DistL2 {}, path, default_ann_params())
            .unwrap();
    let base_bytes = base.to_bytes();

    let mut old_bytes = Vec::new();
    old_bytes.push(1u8); // has_base
    old_bytes.extend_from_slice(&(base_bytes.len() as u64).to_le_bytes());
    old_bytes.extend_from_slice(&base_bytes);
    old_bytes.extend_from_slice(&(16u64).to_le_bytes()); // dim
    old_bytes.extend_from_slice(&0u64.to_le_bytes()); // num_delta = 0
    old_bytes.extend_from_slice(&0u64.to_le_bytes()); // num_graph = 0
    old_bytes.extend_from_slice(&(-1i64).to_le_bytes()); // entry_point = None
    old_bytes.extend_from_slice(&(32u64).to_le_bytes()); // max_degree
    old_bytes.extend_from_slice(&0u64.to_le_bytes()); // num_tombstones = 0

    let loaded = IncrementalDiskANN::<DistL2>::from_bytes(
        &old_bytes,
        DistL2 {},
        IncrementalConfig::default(),
    )
    .unwrap();

    assert_eq!(loaded.stats().base_vectors, 30);
    assert!(!loaded.has_labels());
    assert!(!loaded.has_quantizer());

    // Search should work
    let results = loaded.search(&vectors[0], 5, 32);
    assert_eq!(results.len(), 5);

    cleanup(&[path]);
}

// =========================================================================
// 18. Multi-Seed Beam Search Test
// =========================================================================

#[test]
fn test_beam_search_multi_seed() {
    let path = "test_multi_seed.db";
    cleanup(&[path]);

    // Build an index and search with it
    let vectors = random_vectors(100, 16, 1800);
    let index = IncrementalDiskANN::<DistL2>::build_default(&vectors, path).unwrap();

    // Add a delta so we have two components
    let delta_vecs = random_vectors(20, 16, 1801);
    index.add_vectors(&delta_vecs).unwrap();

    // Search should find results from both base and delta
    let results = index.search_with_dists(&delta_vecs[0], 10, 64);
    assert_eq!(results.len(), 10);

    // Distances should be sorted
    for pair in results.windows(2) {
        assert!(pair[0].1 <= pair[1].1 + 1e-6);
    }

    // No duplicate IDs
    let ids: HashSet<u64> = results.iter().map(|(id, _)| *id).collect();
    assert_eq!(ids.len(), results.len());

    cleanup(&[path]);
}

#[test]
fn test_incremental_search_regression() {
    let path = "test_incr_regression.db";
    cleanup(&[path]);

    let dim = 32;
    let n = 200;
    let vectors = random_vectors(n, dim, 1900);
    let index = IncrementalDiskANN::<DistL2>::build_default(&vectors, path).unwrap();

    // Add delta vectors
    let delta_vecs = random_vectors(50, dim, 1901);
    index.add_vectors(&delta_vecs).unwrap();

    // Delete some base vectors
    index.delete_vectors(&[0, 1, 2, 3, 4]).unwrap();

    // Test recall: search for known vectors
    let mut found_base = 0;
    let mut found_delta = 0;

    for i in 5..20 {
        let results = index.search(&vectors[i], 1, 64);
        if !results.is_empty() && results[0] == i as u64 {
            found_base += 1;
        }
    }

    for v in delta_vecs.iter().take(10) {
        let results = index.search(v, 1, 64);
        if !results.is_empty() {
            // The closest result should be very close to the query
            let best_vec = index.get_vector(results[0]).unwrap();
            let dist: f32 = v
                .iter()
                .zip(&best_vec)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
                .sqrt();
            if dist < 0.01 {
                found_delta += 1;
            }
        }
    }

    assert!(
        found_base >= 10,
        "Should find at least 10/15 base vectors, found {}",
        found_base
    );
    assert!(
        found_delta >= 5,
        "Should find at least 5/10 delta vectors, found {}",
        found_delta
    );

    cleanup(&[path]);
}
