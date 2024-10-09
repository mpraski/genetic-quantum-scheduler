use std::collections::HashSet;
use crate::genetic_algo::{Chromosome, Config, dependencies, evolve, execution_times, generate_population, initial_waiting_times, topological_sort, visualize_chromsome, visualize_schedule};
use rand::prelude::*;

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

fn main() {
    let jobs: u32 = 10_000;
    let backends: u32 = 500;
    let dep_count = 500;

    let execution_times = execution_times(jobs as usize, backends as usize);
    let waiting_times = initial_waiting_times(backends as usize);
    let dependencies = dependencies(jobs as usize, dep_count);
    let topological_orders = generate_topological_orders(&dependencies, jobs, 100);

    let mut deps_hash = HashSet::new();
    for &(dep_job, dependent) in dependencies.iter() {
        deps_hash.insert(dep_job);
        deps_hash.insert(dependent);
    }

    let pop_size = 10_000;
    let sel_size = 2000;
    let elitism_size = (sel_size as f32 * 0.1).trunc() as usize;
    let selection_size = sel_size - elitism_size;
    let mutation_pairs = (jobs as f32 * 0.1).trunc() as usize;
    let mutation_rate = 0.25;

    let config = Config {
        jobs,
        backends,
        deps_hash,
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
    };

    let mut best_fitness: f32 = 0.0;
    let mut best_fitness_count: u32 = 0;

    let population = evolve(generate_population(pop_size, &config), &config, |population: &[Chromosome], generation: i32| -> bool {
        if let Some(best) = population.first() {
            if best_fitness < best.fitness {
                best_fitness = best.fitness;
                best_fitness_count = 0;
            } else {
                best_fitness_count += 1;
            }
        }

        generation == 100 || best_fitness_count == 10
    });


    let schedule = visualize_chromsome(population.first().unwrap(), &config);
    let _visualised = visualize_schedule(&schedule, &config, "out.png");

    println!("Hello, world!");
}
