// examples/demo.rs
use diskann_rs::{DiskAnnError, DistCosine, SPFresh, SPFreshConfig};
use rand::prelude::*;

fn main() -> Result<(), DiskAnnError> {
    let index_path = "spfresh_demo";
    let num_vectors = 1_000_000usize;
    let dim = 1024usize;

    // Build-time knobs (see SPFreshConfig)
    let cfg = SPFreshConfig {
        max_posting_size: 128,
        ..Default::default()
    };

    // Build if missing (SPFresh writes {index_path}.spf/.postings/.vectors/.centroids)
    if !std::path::Path::new(&format!("{index_path}.spf")).exists() {
        println!("Building SPFresh index at {index_path}...");

        // Generate sample vectors (replace with your real dataset)
        println!("Generating {num_vectors} sample vectors of dimension {dim}...");
        let mut rng = thread_rng();
        let vectors: Vec<Vec<f32>> = (0..num_vectors)
            .map(|_| (0..dim).map(|_| rng.r#gen::<f32>()).collect())
            .collect();

        // Pass Some(QuantizerKind::...) to store quantized postings
        let index = SPFresh::<DistCosine>::build(&vectors, index_path, cfg, None)?;
        let stats = index.stats();
        println!(
            "Build done. Index contains {} vectors (dim={}, postings={})",
            stats.live,
            index.dim(),
            stats.postings
        );
    } else {
        println!("Index {index_path} already exists, skipping build.");
    }

    // Open the index (distance type must match what you used to build)
    let index = SPFresh::<DistCosine>::open(index_path)?;
    let stats = index.stats();
    println!(
        "Opened index: {} vectors, dimension={}, postings={}",
        stats.live,
        index.dim(),
        stats.postings
    );

    // Perform a sample query
    let mut rng = thread_rng();
    let query: Vec<f32> = (0..index.dim()).map(|_| rng.r#gen::<f32>()).collect();

    let k = 10usize;
    let n_probe = 8usize;

    println!("\nSearching for {k} nearest neighbors with n_probe={n_probe}...");
    let start = std::time::Instant::now();
    let neighbors: Vec<u64> = index.search(&query, k, n_probe);
    let elapsed = start.elapsed();

    println!("Search completed in {:?}", elapsed);
    println!("Found {} neighbors:", neighbors.len());
    for (i, &id) in neighbors.iter().enumerate() {
        println!("  {}: id {}", i + 1, id);
    }

    Ok(())
}
