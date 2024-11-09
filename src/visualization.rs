use crate::scheduling_optimizer::{Config, ScheduleChromosome};
use plotters::prelude::*;
use plotters::prelude::{Color, Palette99, BLACK, RED, WHITE};
use plotters::style::full_palette::GREY;
use std::error::Error;

pub fn visualize_chromsome(
    chromosome: &ScheduleChromosome,
    config: &Config,
) -> Vec<(u32, u32, u32, u32)> {
    let mut schedule: Vec<(u32, u32, u32, u32)> = Vec::with_capacity(config.jobs as usize);
    let mut job_end_times: Vec<u32> = vec![0; config.jobs as usize];
    let mut last_end_times: Vec<u32> = vec![0; config.backends as usize];

    for &(job, backend) in chromosome.genes.iter() {
        let mut start_time: u32 = 0;

        if config.waiting_times[backend as usize] > start_time {
            start_time = config.waiting_times[backend as usize];
        }

        if last_end_times[backend as usize] > start_time {
            start_time = last_end_times[backend as usize];
        }

        for &(dep_job, dependent) in config.dependencies.iter() {
            if dependent == job {
                let dep_end_time = job_end_times[dep_job as usize];
                if dep_end_time > start_time {
                    start_time = dep_end_time;
                }
            }
        }

        let execution_time =
            config.execution_times[job as usize * config.backends as usize + backend as usize];
        let makespan = start_time + execution_time;

        last_end_times[backend as usize] = makespan;

        job_end_times[job as usize] = makespan;

        schedule.push((job, backend, start_time, start_time + execution_time));
    }

    schedule
}

pub fn visualize_schedule(
    schedule: &[(u32, u32, u32, u32)],
    config: &Config,
    visualize_deps: bool,
    output_path: &str,
) -> Result<(), Box<dyn Error>> {
    // Create a drawing area for the chart.
    let root = BitMapBackend::new(output_path, (2000, 2000)).into_drawing_area();
    root.fill(&WHITE)?;

    // Determine the range of backends and time.
    let max_backend = schedule
        .iter()
        .map(|&(_, backend, _, _)| backend)
        .max()
        .unwrap_or(0);
    let max_time = schedule
        .iter()
        .map(|&(_, _, _, end_time)| end_time)
        .max()
        .unwrap_or(0);

    // Define the chart area and margins.
    let mut chart = ChartBuilder::on(&root)
        .caption("Job Schedule", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0..max_time, 0..max_backend + 1)?;

    // Configure the chart's x and y labels.
    chart
        .configure_mesh()
        .x_desc("Time")
        .y_desc("Backend")
        .y_labels(max_backend as usize + 1)
        .x_labels(10)
        .draw()?;

    for (b, &t) in config.waiting_times.iter().enumerate() {
        let color = &GREY;
        chart
            .draw_series(vec![Rectangle::new(
                [(0, b as u32), (t, (b + 1) as u32)], // The rectangle spans the time and backend range.
                color.filled(),
            )])?
            .label(format!("Backend {}", b))
            .legend(move |(x, y)| Rectangle::new([(x, y - 5), (x + 10, y + 5)], color.filled()));
    }

    // Draw each job as a bar in the chart.
    for &(job, backend, start_time, end_time) in schedule {
        let color = if visualize_deps {
            if config.deps_hash.contains(&job) {
                RED.to_rgba()
            } else {
                Palette99::pick(job as usize).mix(0.25)
            }
        } else {
            Palette99::pick(job as usize).to_rgba()
        };

        chart
            .draw_series(vec![Rectangle::new(
                [(start_time, backend), (end_time, backend + 1)], // The rectangle spans the time and backend range.
                color.filled(),
            )])?
            .label(format!("Job {}", job))
            .legend(move |(x, y)| Rectangle::new([(x, y - 5), (x + 10, y + 5)], color.filled()));
    }

    // Configure the legend for the chart.
    chart
        .configure_series_labels()
        .border_style(&BLACK)
        .draw()?;

    // Save the result to the specified output path.
    root.present()?;
    println!("Chart saved to {}", output_path);
    Ok(())
}
