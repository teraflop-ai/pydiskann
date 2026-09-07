// examples/diskann_mnist.rs
#![allow(clippy::needless_range_loop)]

use cpu_time::ProcessTime;
use diskann_rs::{DiskAnnError, DistL2, SPFresh, SPFreshConfig};
use rayon::prelude::*;
use std::time::{Duration, SystemTime};
mod utils;
use utils::*;

fn euclid(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for j in 0..a.len() {
        let d = a[j] - b[j];
        s += d * d;
    }
    s.sqrt()
}

fn main() -> Result<(), DiskAnnError> {
    // Load ANN benchmark data (Fashion-MNIST, L2, HDF5), must be in crate root
    // wget http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5
    let fname = String::from("./fashion-mnist-784-euclidean.hdf5");
    println!("\n\nSPFresh benchmark on {:?}", fname);

    let anndata = annhdf5::AnnBenchmarkData::new(fname.clone())
        .expect("Failed to load fashion-mnist-784-euclidean.hdf5");
    let knbn_max = anndata.test_distances.dim().1;
    let nb_elem = anndata.train_data.len();
    let nb_search = anndata.test_data.len();

    println!("Train size : {}", nb_elem);
    println!("Test size  : {}", nb_search);
    println!("Ground-truth k per query in file: {}", knbn_max);

    // SPFresh build parameters (tune as desired)
    let max_posting_size = 128; // split postings at this size
    let search_k = 10; // evaluate @k=10 (matches HNSW example)
    let n_probe = 32; // postings scanned per query: speed/recall tradeoff

    // Build vectors for SPFresh (we only need the float rows; ids are implicit 0..n-1)
    // anndata.train_data is Vec<(Vec<f32>, usize_id)>; we only take the Vec<f32> in order.
    let train_vectors: Vec<Vec<f32>> = anndata
        .train_data
        .iter()
        .map(|pair| pair.0.clone())
        .collect();

    // Build (if needed) or open index
    let index_path = "spfresh_mnist";
    let index = if !std::path::Path::new(&format!("{index_path}.spf")).exists() {
        println!(
            "\nBuilding SPFresh index: n={}, dim={}, max_posting_size={}",
            train_vectors.len(),
            train_vectors[0].len(),
            max_posting_size
        );

        let cfg = SPFreshConfig {
            max_posting_size,
            ..Default::default()
        };

        let start_cpu = ProcessTime::now();
        let start_wall = SystemTime::now();

        let idx = SPFresh::<DistL2>::build(&train_vectors, index_path, cfg, None)?;

        let cpu_time: Duration = start_cpu.elapsed();
        let wall_time = start_wall.elapsed().unwrap();
        println!(
            "Build complete. CPU time: {:?}, wall time: {:?}",
            cpu_time, wall_time
        );

        idx
    } else {
        println!("\nIndex {} exists, opening…", index_path);
        let start_wall = SystemTime::now();
        let idx = SPFresh::<DistL2>::open(index_path)?;
        let wall_time = start_wall.elapsed().unwrap();
        println!(
            "Opened index: {} vectors, dim={} in {:?}",
            idx.stats().live,
            idx.dim(),
            wall_time
        );
        idx
    };

    // Search (parallel), compute recall
    println!(
        "\nSearching {} queries with k={}, n_probe={} …",
        nb_search, search_k, n_probe
    );

    // Parallel search timing
    let start_cpu = ProcessTime::now();
    let start_wall = SystemTime::now();

    // For each test vector, run SPFresh search, then compute distances of returned neighbors
    // to compare vs. ground truth threshold (k-th smallest true distance).
    let results_dists: Vec<Vec<f32>> = anndata
        .test_data
        .par_iter()
        .map(|q| {
            let ids = index.search(q, search_k, n_probe);
            // compute distances for the returned neighbors
            let mut ds = Vec::with_capacity(ids.len());
            for &id in &ids {
                let v = index.get_vector(id).expect("missing vector");
                ds.push(euclid(q, &v));
            }
            ds.sort_by(|a, b| a.partial_cmp(b).unwrap());
            ds
        })
        .collect();

    let cpu_time = start_cpu.elapsed();
    let wall_time = start_wall.elapsed().unwrap();

    // Metrics (same style as HNSW example)
    // - recall: fraction of returned neighbors whose distance ≤ true k-th distance
    // - mean fraction nb returned: (#returned) / (k * #queries)
    // - last distances ratio: avg( last_returned_distance / true_kth_distance )
    // - req/s from wall time
    let mut recalls: Vec<usize> = Vec::with_capacity(nb_search);
    let mut nb_returned: Vec<usize> = Vec::with_capacity(nb_search);
    let mut last_distances_ratio: Vec<f32> = Vec::with_capacity(nb_search);

    for i in 0..nb_search {
        let true_row = anndata.test_distances.row(i);
        // true_row is sorted; k-th NN threshold:
        let true_k = search_k.min(true_row.len());
        let gt_kth = true_row[true_k - 1];

        let dists = &results_dists[i];
        nb_returned.push(dists.len());

        // recall by threshold (same heuristic as HNSW example)
        let recall = dists.iter().filter(|x| **x <= gt_kth).count();
        recalls.push(recall);

        let ratio = if !dists.is_empty() {
            dists[dists.len() - 1] / gt_kth
        } else {
            0.0
        };
        last_distances_ratio.push(ratio);
    }

    let knbn = search_k;
    let mean_recall = (recalls.iter().sum::<usize>() as f32) / ((knbn * recalls.len()) as f32);
    let mean_frac_returned =
        (nb_returned.iter().sum::<usize>() as f32) / ((nb_returned.len() * knbn) as f32);
    let mean_last_ratio =
        last_distances_ratio.iter().sum::<f32>() / (last_distances_ratio.len() as f32);

    let search_sys_time_us = wall_time.as_micros() as f32;
    let req_per_s = (nb_search as f32) * 1.0e6_f32 / search_sys_time_us;

    println!(
        "\n mean fraction nb returned by search {:?}",
        mean_frac_returned
    );
    println!("\n last distances ratio {:?}", mean_last_ratio);
    println!(
        "\n recall rate for {:?} is {:?} , nb req /s {:?}",
        anndata.fname, mean_recall, req_per_s
    );

    println!(
        "\n total cpu time for search requests {:?} , system time {:?}",
        cpu_time, wall_time
    );

    Ok(())
}
