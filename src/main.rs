use crate::demo_data::{
    dependencies, execution_times, fidelities, initial_waiting_times, topological_sort,
};
use crate::genetic_algorithm::{Chromosome, Evaluator, Optimizer};
use crate::genetic_optimizer::{
    GeneticOptimizer, QuantumSchedule, SchedulingAlgorithm, SchedulingConfig, SchedulingEvaluator,
};
use crate::nsga2_optimizer::NSGA2Optimizer;
use chrono::Local;
use csv::Writer;
use itertools::iproduct;
use rand::prelude::*;
use serde::Serialize;
use std::cmp;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashSet;
use std::error::Error;
use std::fs::OpenOptions;
use std::hash::{Hash, Hasher};
use std::iter::Sum;
use std::time::Instant;

mod demo_data;
mod dominance_ord;
mod genetic_algorithm;
mod genetic_optimizer;
mod non_dominated_sort;
mod nsga2_optimizer;
mod visualization;

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
    multi_objective_ratio: Vec<f64>,
}

#[derive(Debug, Serialize)]
pub struct FinalTestResult {
    pub scenario: u64,
    pub repetitions: i32,
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
    pub mean_makespan: f64,
    pub mean_fidelity: f64,
    pub mean_runtime: f64,
    pub mean_generations: f64,
    pub var_fitness: f64,
    pub var_makespan: f64,
    pub var_fidelity: f64,
    pub var_runtime: f64,
    pub var_generations: f64,
    pub selection_size_percentage: f64,
    pub elitism_size_percentage: f64,
    pub mutation_pairs_percentage: f64,
    pub mutation_rate_percentage: f64,
    pub dependency_percentage: f64,
    pub multi_objective_ratio: f64,
}

#[derive(Debug)]
pub struct RunResult<C: Chromosome> {
    runtime: f64,
    generations: i32,
    best_specimen: C,
}

const REPETITIONS: i32 = 5;

fn mean_variance<T: Copy + Into<f64> + Sum<T>>(values: &[T]) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }

    let n = values.len() as f64;
    let sum: f64 = values.iter().map(|&v| v.into()).sum();
    let mean = sum / n;

    let variance = values
        .iter()
        .map(|&v| {
            let diff = v.into() - mean;
            diff * diff
        })
        .sum::<f64>()
        / n;

    (mean, variance)
}

fn hash_combination(
    nsga2: &bool,
    jobs: &u32,
    backends: &u32,
    selection_size_p: &f64,
    elitism_size_p: &f64,
    mutation_pairs_p: &f64,
    mutation_rate_p: &f64,
    dependency: &f64,
    multi_objective_ratio: &f64,
) -> u64 {
    let mut hasher = DefaultHasher::new();

    nsga2.hash(&mut hasher);
    jobs.hash(&mut hasher);
    backends.hash(&mut hasher);
    selection_size_p.to_bits().hash(&mut hasher); // Hashing the f32 as its bit representation
    elitism_size_p.to_bits().hash(&mut hasher);
    mutation_pairs_p.to_bits().hash(&mut hasher);
    mutation_rate_p.to_bits().hash(&mut hasher);
    dependency.to_bits().hash(&mut hasher);
    multi_objective_ratio.to_bits().hash(&mut hasher);

    hasher.finish()
}

fn collect_benchmarks(schemas: &[TestSchema], file_path: &str) -> Result<(), Box<dyn Error>> {
    let mut visited: HashSet<u64> = HashSet::new();
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(file_path)?;
    let mut writer = Writer::from_writer(file);

    for schema in schemas {
        for (
            &nsga2,
            &jobs,
            &backends,
            &selection_size_p,
            &elitism_size_p,
            &mutation_pairs_p,
            &mutation_rate_p,
            &dependency,
            &multi_objective_ratio,
        ) in iproduct!(
            &schema.nsga2,
            &schema.jobs,
            &schema.backends,
            &schema.selection_size_percentage,
            &schema.elitism_size_percentage,
            &schema.mutation_pairs_percentage,
            &schema.mutation_rate_percentage,
            &schema.dependency_percentage,
            &schema.multi_objective_ratio
        ) {
            let hash = hash_combination(
                &nsga2,
                &jobs,
                &backends,
                &selection_size_p,
                &elitism_size_p,
                &mutation_pairs_p,
                &mutation_rate_p,
                &dependency,
                &multi_objective_ratio,
            );

            if visited.contains(&hash) {
                println!("Scenario {} already evaluated, skipping...", hash);
                continue;
            } else {
                println!("Scenario {} is being run...", hash);
            }

            visited.insert(hash);

            let fidels = fidelities(jobs as usize, backends as usize);
            let execution_times = execution_times(jobs as usize, backends as usize);
            let waiting_times = initial_waiting_times(backends as usize);
            let dependencies =
                dependencies(jobs as usize, (jobs as f64 * dependency).trunc() as usize);
            let topological_orders = generate_topological_orders(&dependencies, jobs, 200);

            let mut deps_hash = HashSet::new();
            for &(dep_job, dependent) in dependencies.iter() {
                deps_hash.insert(dep_job);
                deps_hash.insert(dependent);
            }

            let population_size = cmp::max(10_000, jobs as usize * 5);
            let sel_size = cmp::max(2_000, (jobs as f64 * selection_size_p).trunc() as usize);
            let elitism_size = (sel_size as f64 * elitism_size_p).trunc() as usize;
            let selection_size = sel_size - elitism_size;
            let mutation_pairs = (jobs as f64 * mutation_pairs_p).trunc() as usize;

            let config = SchedulingConfig {
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
                multi_objective_ratio,
            };

            let algorithm = Box::new(SchedulingAlgorithm {
                config: config.clone(),
            });
            let mut optimizer: Box<dyn Optimizer<QuantumSchedule>> = if nsga2 {
                Box::new(NSGA2Optimizer { algorithm })
            } else {
                Box::new(GeneticOptimizer { algorithm })
            };

            let mut runs = Vec::with_capacity(REPETITIONS as usize);
            for i in 0..REPETITIONS {
                println!("-- Repetition {} of {} is being run...", i, hash);
                runs.push(benchmark_run(&mut optimizer))
            }

            let fitness_values: Vec<f64> = runs.iter().map(|r| r.best_specimen.fitness).collect();
            let makespan_values: Vec<u32> = runs.iter().map(|r| r.best_specimen.makespan).collect();
            let fidelity_values: Vec<f64> =
                runs.iter().map(|r| r.best_specimen.mean_fidelity).collect();
            let runtime_values: Vec<f64> = runs.iter().map(|r| r.runtime).collect();
            let generations_values: Vec<i32> = runs.iter().map(|r| r.generations).collect();

            let (mean_fitness, var_fitness) = mean_variance(&fitness_values);
            let (mean_makespan, var_makespan) = mean_variance(&makespan_values);
            let (mean_fidelity, var_fidelity) = mean_variance(&fidelity_values);
            let (mean_runtime, var_runtime) = mean_variance(&runtime_values);
            let (mean_generations, var_generations) = mean_variance(&generations_values);

            let result = FinalTestResult {
                scenario: hash,
                repetitions: REPETITIONS,
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
                makespan_weight: multi_objective_ratio,
                fidelity_weight: 1.0 - multi_objective_ratio,
                mean_fitness,
                mean_makespan,
                mean_fidelity,
                mean_runtime,
                mean_generations,
                var_fitness,
                var_makespan,
                var_fidelity,
                var_runtime,
                var_generations,
                selection_size_percentage: selection_size_p,
                elitism_size_percentage: elitism_size_p,
                mutation_pairs_percentage: mutation_pairs_p,
                mutation_rate_percentage: mutation_rate_p,
                dependency_percentage: dependency,
                multi_objective_ratio,
            };

            writer.serialize(result)?;
            writer.flush()?;
        }
    }

    Ok(())
}

fn benchmark_run(
    optimizer: &mut Box<dyn Optimizer<QuantumSchedule>>,
) -> RunResult<QuantumSchedule> {
    let mut evaluator: Box<dyn Evaluator<QuantumSchedule>> = Box::new(SchedulingEvaluator {
        ..Default::default()
    });

    let start = Instant::now();
    let population = optimizer.optimize(&mut evaluator);
    let runtime = start.elapsed().as_secs_f64();

    RunResult {
        runtime,
        generations: evaluator.generation(),
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

fn main() {
    let schemas = vec![
        TestSchema {
            nsga2: vec![false, true],
            jobs: vec![128, 256, 512],
            backends: vec![16, 32],
            selection_size_percentage: vec![0.25],
            elitism_size_percentage: vec![0.05],
            mutation_pairs_percentage: vec![0.15],
            mutation_rate_percentage: vec![0.15],
            dependency_percentage: vec![0.1, 0.2, 0.3],
            multi_objective_ratio: vec![0.0, 0.25, 0.5, 0.75, 1.0],
        },
        TestSchema {
            nsga2: vec![false, true],
            jobs: vec![512, 1024, 2048, 4096],
            backends: vec![32, 64],
            selection_size_percentage: vec![0.25],
            elitism_size_percentage: vec![0.05],
            mutation_pairs_percentage: vec![0.15],
            mutation_rate_percentage: vec![0.15],
            dependency_percentage: vec![0.1, 0.2, 0.3],
            multi_objective_ratio: vec![0.0, 0.25, 0.5, 0.75, 1.0],
        },
        TestSchema {
            nsga2: vec![false, true],
            jobs: vec![2048, 4096, 8192, 16384],
            backends: vec![64, 128],
            selection_size_percentage: vec![0.25],
            elitism_size_percentage: vec![0.05],
            mutation_pairs_percentage: vec![0.15],
            mutation_rate_percentage: vec![0.15],
            dependency_percentage: vec![0.1, 0.2, 0.3],
            multi_objective_ratio: vec![0.0, 0.25, 0.5, 0.75, 1.0],
        },
    ];

    let now = Local::now();
    let date_str = now.format("%Y-%m-%d_%H-%M-%S").to_string();
    let filename = format!("test_results_{}.csv", date_str);

    collect_benchmarks(&schemas, &filename).unwrap();
}
