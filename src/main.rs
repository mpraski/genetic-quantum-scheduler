use std::collections::HashSet;
use crate::genetic_algo::{Chromosome, Config, fidelities, dependencies, evolve, execution_times, generate_population, initial_waiting_times, topological_sort, visualize_chromsome, visualize_schedule, evolve_nsga2};
use rand::prelude::*;
use std::time::{Instant};
use colored::Colorize;
use std::cmp;

mod genetic_algo;

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

fn test_run(config: &Config, pop_size: usize, nsga2: bool, name: &str) {
    let chosen_function = if nsga2 {
        evolve_nsga2
    } else {
        evolve
    };

    let mut best_fitness: f64 = 0.0;
    let mut best_fitness_count: u32 = 0;

    let start = Instant::now();
    let population = chosen_function(generate_population(pop_size, &config), &config, |population: &[Chromosome], generation: i32| -> bool {
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
    let jobs: u32 = 10_000; // 1,000 - 20,000
    let backends: u32 = 100; // 16, 32, 64, 128
    let dep_count = 100; // to-do, 10-20% of jobs

    let fidels = fidelities(jobs as usize, backends as usize);
    let execution_times = execution_times(jobs as usize, backends as usize);
    let waiting_times = initial_waiting_times(backends as usize);
    let dependencies = dependencies(jobs as usize, dep_count);
    let topological_orders = generate_topological_orders(&dependencies, jobs, 200);

    let mut deps_hash = HashSet::new();
    for &(dep_job, dependent) in dependencies.iter() {
        deps_hash.insert(dep_job);
        deps_hash.insert(dependent);
    }

    let pop_size = cmp::max(50_000, jobs as usize * 3);
    let sel_size = 2_000;
    let elitism_size = (sel_size as f32 * 0.1).trunc() as usize;
    let selection_size = sel_size - elitism_size;
    let mutation_pairs = (jobs as f32 * 0.15).trunc() as usize;
    let mutation_rate = 0.15;

    let config = Config {
        jobs,
        backends,
        deps_hash,
        fidelities: fidels,
        dependencies,
        waiting_times,
        execution_times,
        topological_orders,
        elitism_size,
        selection_size,
        mutation_rate,
        mutation_pairs,
        gap_threshold: 10,
        max_utilized_backends: (jobs as f32 * 0.05).trunc() as usize,
        max_underutilized_backends: (jobs as f32 * 0.15).trunc() as usize,
        makespan_weight: 0.5,
        fidelity_weight: 0.5,
    };

    test_run(&config, pop_size, false, "vanilla");
    test_run(&config, pop_size, false, "nsga2");
}
