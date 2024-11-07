use std::collections::HashMap;
use petgraph::algo::toposort;
use petgraph::graph::{DiGraph, NodeIndex};
use rand::{thread_rng, Rng};
use rand::prelude::SliceRandom;
use rand_distr::{Normal, Distribution};

pub fn execution_times(jobs: usize, backends: usize) -> Vec<u32> {
    (0..jobs)
        .flat_map(|_| {
            (0..backends)
                .map(|_| thread_rng().gen_range(5..=20))
                .collect::<Vec<u32>>()
        })
        .collect()
}

pub fn initial_waiting_times(backends: usize) -> Vec<u32> {
    (0..backends)
        .map(|_| thread_rng().gen_range(5..=20))
        .collect()
}

pub fn fidelities(jobs: usize, backends: usize) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    let mut fidelities: Vec<f64> = Vec::with_capacity(jobs * backends);
    let mut distribution: Vec<Normal<f64>> = Vec::with_capacity(backends);

    // Have decreasing order of mean fidelity
    // Keep the mean variable
    for _ in 0..backends {
        let mean = rng.gen_range(0.0..=1.0);
        let std_dev = 1.0;

        distribution.push(Normal::new(mean, std_dev).unwrap());
    }

    for _ in 0..jobs {
        for d in distribution.iter_mut() {
            fidelities.push(d.sample(&mut rng).clamp(0.0, 1.0))
        }
    }

    fidelities
}

pub fn dependencies(jobs: usize, deps_pairs: usize) -> Vec<(u32, u32)> {
    let mut rng = thread_rng();
    let mut job_indices: Vec<u32> = (0..jobs as u32).collect();

    job_indices.shuffle(&mut rng);

    let pairs: Vec<(u32, u32)> = job_indices
        .windows(2)
        .map(|window| (window[0], window[1]))
        .filter(|&(a, b)| a < b) // Keep only valid pairs where a < b to avoid cyclic dependencies.
        .collect();

    pairs
        .choose_multiple(&mut rng, deps_pairs)
        .cloned()
        .collect()
}

pub fn topological_sort(jobs: &[u32], dependencies: &[(u32, u32)]) -> Option<Vec<u32>> {
    let mut graph = DiGraph::<u32, ()>::new();
    let mut job_indices = HashMap::new();

    let data = prepare(jobs, dependencies);

    for &(job, ref deps) in data.iter() {
        let job_index = *job_indices.entry(job).or_insert_with(|| graph.add_node(job));

        for &dep in deps {
            add_dependency(&mut graph, &mut job_indices, job_index, dep);
        }
    }

    match toposort(&graph, None) {
        Ok(sorted) => Some(sorted.iter().map(|&idx| graph[idx]).collect()),
        Err(_) => None,
    }
}

fn add_dependency(
    graph: &mut DiGraph<u32, ()>,
    job_indices: &mut HashMap<u32, NodeIndex>,
    job_index: NodeIndex,
    dep: u32,
) {
    // Add the dependency node if it does not exist
    let dep_index = *job_indices.entry(dep).or_insert_with(|| graph.add_node(dep));

    // Add an edge from the dependency to the job
    graph.add_edge(dep_index, job_index, ());
}

fn prepare(jobs: &[u32], dependencies: &[(u32, u32)]) -> Vec<(u32, Vec<u32>)> {
    let mut deps_map: HashMap<u32, Vec<u32>> = HashMap::new();

    for &(job, dep) in dependencies {
        deps_map.entry(dep).or_insert_with(Vec::new).push(job);
    }

    jobs.iter()
        .map(|&job| (job, deps_map.get(&job).cloned().unwrap_or_default()))
        .collect()
}
