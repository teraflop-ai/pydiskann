//! Generate quantization comparison SVG chart
//!
//! Run with: cargo run --example generate_quantization_chart --release

use plotters::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let output_path = "docs/charts/quantization_comparison.svg";

    // Data: (method, compression_ratio, recall%)
    let data = vec![
        ("None", 1.0f32, 100.0f32),
        ("F16", 2.0, 100.0),
        ("Int8", 4.0, 99.0),
        ("PQ-32", 16.0, 62.0),
        ("PQ-16", 32.0, 30.0),
        ("PQ-8", 64.0, 12.0),
    ];

    let root = SVGBackend::new(output_path, (600, 400)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Quantization: Compression vs Recall@10", ("sans-serif", 22))
        .margin(15)
        .x_label_area_size(45)
        .y_label_area_size(55)
        .build_cartesian_2d((0.8f32..80f32).log_scale(), 0f32..105f32)?;

    chart
        .configure_mesh()
        .x_desc("Compression Ratio (log scale)")
        .y_desc("Recall@10 (%)")
        .x_label_formatter(&|x| format!("{}x", *x as i32))
        .draw()?;

    // Colors for each method
    let colors = [
        RGBColor(100, 100, 100), // None - gray
        RGBColor(52, 152, 219),  // F16 - blue
        RGBColor(46, 204, 113),  // Int8 - green
        RGBColor(155, 89, 182),  // PQ-32 - purple
        RGBColor(230, 126, 34),  // PQ-16 - orange
        RGBColor(231, 76, 60),   // PQ-8 - red
    ];

    // Draw connecting line (gray)
    chart.draw_series(LineSeries::new(
        data.iter().map(|(_, x, y)| (*x, *y)),
        &RGBColor(180, 180, 180),
    ))?;

    // Draw points and labels
    for (i, (label, x, y)) in data.iter().enumerate() {
        // Draw circle
        chart
            .draw_series(std::iter::once(Circle::new(
                (*x, *y),
                6,
                colors[i].filled(),
            )))?
            .label(*label)
            .legend(move |(lx, ly)| Circle::new((lx + 10, ly), 5, colors[i].filled()));
    }

    // Draw legend
    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .margin(10)
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK.mix(0.3))
        .draw()?;

    root.present()?;
    println!("Chart saved to {}", output_path);

    Ok(())
}
