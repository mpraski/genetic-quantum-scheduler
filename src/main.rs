use std::collections::HashSet;
use crate::genetic_algo::{Chromosome, Config, evolve, generate_population, visualize_chromsome, visualize_schedule, ScheduleChromosome};
use rand::prelude::*;
use std::time::{Instant};
use colored::Colorize;
use std::cmp;
use std::error::Error;
use std::fs::File;
use csv::Writer;
use itertools::iproduct;
use serde::Serialize;
use chrono::Local;
use crate::demo_data::{dependencies, execution_times, fidelities, initial_waiting_times, topological_sort};
use crate::nsga2::evolve_nsga2;

mod genetic_algo;
mod dominance_ord;
mod non_dominated_sort;
mod demo_data;
mod nsga2;

#[derive(Debug)]
pub struct TestSchema {
    nsga2: Vec<bool>,
    jobs: Vec<u32>,
    backends: Vec<u32>,
    selection_size_percentage: Vec<f64>,
    elitism_size_percentage: Vec<f64>,
    mutation_pairs_percentage: Vec<f64>,
    mutation_rate_percentage: Vec<f64>,
    dependency_percentage: Vec<f64>,
}

#[derive(Debug, Serialize)]
pub struct FinalTestResult {
    pub nsga2: bool,
    pub jobs: u32,
    pub backends: u32,
    pub population_size: usize,
    pub elitism_size: usize,
    pub selection_size: usize,
    pub mutation_rate: f64,
    pub mutation_pairs: usize,
    pub gap_threshold: usize,
    pub max_utilized_backends: usize,
    pub max_underutilized_backends: usize,
    pub makespan_weight: f64,
    pub fidelity_weight: f64,
    pub mean_fitness: f64,
    pub mean_makespan: u32,
    pub mean_fidelity: f64,
    pub mean_runtime: f64,
    pub mean_generations: u32,
    pub selection_size_percentage: f64,
    pub elitism_size_percentage: f64,
    pub mutation_pairs_percentage: f64,
    pub mutation_rate_percentage: f64,
    pub dependency_percentage: f64,
}

#[derive(Debug)]
pub struct TestResult<C: Chromosome> {
    config: Config,
    runs: Vec<RunResult<C>>,
}

#[derive(Debug)]
pub struct RunResult<C: Chromosome> {
    runtime: f64,
    generations: usize,
    best_specimen: C,
}

const REPETITIONS: i32 = 1;

pub fn write_to_csv(records: &[FinalTestResult], filename: &str) -> Result<(), Box<dyn Error>> {
    let file = File::create(filename)?;
    let mut wtr = Writer::from_writer(file);

    for record in records {
        wtr.serialize(record)?;
    }

    wtr.flush()?;
    Ok(())
}

fn collect_benchmarks(schema: &TestSchema) -> Vec<FinalTestResult> {
    let mut results = Vec::new();

    for (&nsga2, &jobs, &backends, &selection_size_p, &elitism_size_p, &mutation_pairs_p, &mutation_rate_p, &dependency) in
        iproduct!(
                &schema.nsga2,
                &schema.jobs,
                &schema.backends,
                &schema.selection_size_percentage,
                &schema.elitism_size_percentage,
                &schema.mutation_pairs_percentage,
                &schema.mutation_rate_percentage,
                &schema.dependency_percentage
            )
    {
        if results.len() > 50 {
            break;
        }

        let fidels = fidelities(jobs as usize, backends as usize);
        let execution_times = execution_times(jobs as usize, backends as usize);
        let waiting_times = initial_waiting_times(backends as usize);
        let dependencies = dependencies(jobs as usize, (jobs as f64 * dependency).trunc() as usize);
        let topological_orders = generate_topological_orders(&dependencies, jobs, 200);

        let mut deps_hash = HashSet::new();
        for &(dep_job, dependent) in dependencies.iter() {
            deps_hash.insert(dep_job);
            deps_hash.insert(dependent);
        }

        let population_size = cmp::max(10_000, jobs as usize * 5);
        let sel_size = cmp::max(5_000, (jobs as f64 * selection_size_p).trunc() as usize);
        let elitism_size = (sel_size as f64 * elitism_size_p).trunc() as usize;
        let selection_size = sel_size - elitism_size;
        let mutation_pairs = (jobs as f64 * mutation_pairs_p).trunc() as usize;

        let config = Config {
            nsga2,
            jobs,
            backends,
            deps_hash,
            fidelities: fidels,
            dependencies,
            waiting_times,
            execution_times,
            topological_orders,
            population_size,
            elitism_size,
            selection_size,
            mutation_rate: mutation_rate_p,
            mutation_pairs,
            gap_threshold: 10,
            max_utilized_backends: (jobs as f32 * 0.05).trunc() as usize,
            max_underutilized_backends: (jobs as f32 * 0.15).trunc() as usize,
            makespan_weight: 0.5,
            fidelity_weight: 0.5,
        };

        let mut runs = Vec::with_capacity(REPETITIONS as usize);
        for _ in 0..REPETITIONS {
            runs.push(benchmark_run(&config))
        }

        let sum_fitness: f64 = runs.iter().map(|r| r.best_specimen.fitness).sum();
        let sum_makespan: u32 = runs.iter().map(|r| r.best_specimen.makespan).sum();
        let sum_fidelity: f64 = runs.iter().map(|r| r.best_specimen.mean_fidelity).sum();
        let sum_runtime: f64 = runs.iter().map(|r| r.runtime).sum();
        let sum_generations: usize = runs.iter().map(|r| r.generations).sum();

        results.push(FinalTestResult {
            nsga2: config.nsga2,
            jobs: config.jobs,
            backends: config.backends,
            population_size: config.population_size,
            elitism_size: config.elitism_size,
            selection_size: config.selection_size,
            mutation_rate: config.mutation_rate,
            mutation_pairs: config.mutation_pairs,
            gap_threshold: config.gap_threshold,
            max_utilized_backends: config.max_utilized_backends,
            max_underutilized_backends: config.max_underutilized_backends,
            makespan_weight: config.makespan_weight,
            fidelity_weight: config.fidelity_weight,
            mean_fitness: sum_fitness / REPETITIONS as f64,
            mean_makespan: sum_makespan / REPETITIONS as u32,
            mean_fidelity: sum_fidelity / REPETITIONS as f64,
            mean_runtime: sum_runtime / REPETITIONS as f64,
            mean_generations: (sum_generations as f64 / REPETITIONS as f64).round() as u32,
            selection_size_percentage: selection_size_p,
            elitism_size_percentage: elitism_size_p,
            mutation_pairs_percentage: mutation_pairs_p,
            mutation_rate_percentage: mutation_rate_p,
            dependency_percentage: dependency,
        })
    }

    results
}

fn benchmark_run(config: &Config) -> RunResult<ScheduleChromosome> {
    let chosen_function = if config.nsga2 {
        evolve_nsga2
    } else {
        evolve
    };

    let mut best_fitness: f64 = 0.0;
    let mut best_fitness_count: u32 = 0;
    let mut generations = 0;

    let start = Instant::now();
    let population = chosen_function(generate_population(&config), &config, |population: &[Chromosome], generation: i32| -> bool {
        if let Some(best) = population.first() {
            if best_fitness < best.fitness() {
                best_fitness = best.fitness();
                best_fitness_count = 0;
            } else {
                best_fitness_count += 1;
            }
        }

        generations = generation as usize;

        generation == 200 || best_fitness_count == 20
    });
    let runtime = start.elapsed().as_secs_f64();

    RunResult {
        runtime,
        generations,
        best_specimen: population.first().unwrap().clone(),
    }
}

fn generate_topological_orders(dependencies: &[(u32, u32)], jobs: u32, n: usize) -> Vec<Vec<u32>> {
    let mut unique_orders = HashSet::with_capacity(n);
    let mut result = Vec::with_capacity(n);
    let mut rng = thread_rng();

    while unique_orders.len() < n {
        let mut random_range: Vec<u32> = (0..jobs).collect();
        random_range.shuffle(&mut rng);

        let mut random_deps = dependencies.to_vec(); // Clone the input to create a new vector.
        random_deps.shuffle(&mut rng);

        if let Some(sorted) = topological_sort(&random_range, &random_deps) {
            if unique_orders.insert(sorted.clone()) {
                result.push(sorted);
            }
        }
    }

    result
}

fn test_run(config: &Config, nsga2: bool, name: &str) {
    let chosen_function = if nsga2 {
        evolve_nsga2
    } else {
        evolve
    };

    let mut best_fitness: f64 = 0.0;
    let mut best_fitness_count: u32 = 0;

    let start = Instant::now();
    let population = chosen_function(generate_population(&config), &config, |population: &[Chromosome], generation: i32| -> bool {
        if let Some(best) = population.first() {
            if best_fitness < best.fitness {
                best_fitness = best.fitness;
                best_fitness_count = 0;
            } else {
                best_fitness_count += 1;
            }
        }

        generation == 200 || best_fitness_count == 20
    });
    let duration = start.elapsed().as_secs_f64();
    println!("{}: Took {}", name, format!("{:.2} seconds", duration).bold().green());

    let schedule = visualize_chromsome(population.first().unwrap(), &config);
    let _visualised = visualize_schedule(&schedule, &config, false, &format!("out-{}.png", name));
    let _visualised = visualize_schedule(&schedule, &config, true, &format!("out-{}-deps.png", name));
}

fn main() {
    let schema = TestSchema {
        nsga2: vec![false, true],
        jobs: vec![128, 256, 512, 1024, 2048, 4096, 8192, 16384],
        backends: vec![16, 32, 64, 128],
        selection_size_percentage: vec![0.25],
        elitism_size_percentage: vec![0.1],
        mutation_pairs_percentage: vec![0.15],
        mutation_rate_percentage: vec![0.15],
        dependency_percentage: vec![0.1, 0.15, 0.2],
    };

    let results = collect_benchmarks(&schema);
    let now = Local::now();

    let date_str = now.format("%Y-%m-%d_%H-%M-%S").to_string();
    let filename = format!("test_results_{}_{}.csv", results.len(), date_str);

    write_to_csv(&results, &filename).unwrap();
}
