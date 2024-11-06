use rand::distributions::Uniform;
use rand::prelude::*;
use rayon::prelude::*;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::algo::toposort;
use std::collections::{HashMap, HashSet};
use std::cmp::Ordering;
use plotters::prelude::*;
use std::error::Error;
use plotters::style::full_palette::GREY;
use colored::Colorize;
use rand_distr::{Distribution, Normal};
use std::marker::PhantomData;

#[derive(Clone, Debug, Default)]
pub struct Config {
    pub jobs: u32,
    pub backends: u32,
    pub deps_hash: HashSet<u32>,
    pub fidelities: Vec<f64>,
    pub dependencies: Vec<(u32, u32)>,
    pub waiting_times: Vec<u32>,
    pub execution_times: Vec<u32>,
    pub topological_orders: Vec<Vec<u32>>,
    pub elitism_size: usize,
    pub selection_size: usize,
    pub mutation_rate: f64,
    pub mutation_pairs: usize,
    pub gap_threshold: usize,
    pub max_utilized_backends: usize,
    pub max_underutilized_backends: usize,
    pub makespan_weight: f64,
    pub fidelity_weight: f64,
}

#[derive(Clone, Debug, Default)]
pub struct Chromosome {
    pub genes: Vec<(u32, u32)>,
    pub order: usize,
    pub fitness: f64,
    pub makespan: u32,
    pub mean_fidelity: f64,
}

type SolutionIdx = usize;

pub trait DominanceOrd {
    /// The type on which the dominance relation is defined.
    type T;

    /// Returns the dominance order.
    fn dominance_ord(&self, a: &Self::T, b: &Self::T) -> Ordering {
        if self.dominates(a, b) {
            Ordering::Less
        } else if self.dominates(b, a) {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    }

    /// Returns true if `a` dominates `b` ("a < b").
    fn dominates(&self, a: &Self::T, b: &Self::T) -> bool {
        match self.dominance_ord(a, b) {
            Ordering::Less => true,
            _ => false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Front<'s, S: 's> {
    dominated_solutions: Vec<Vec<SolutionIdx>>,
    domination_count: Vec<usize>,
    previous_front: Vec<SolutionIdx>,
    current_front: Vec<SolutionIdx>,
    rank: usize,
    solutions: &'s [S],
}

pub struct AssignedCrowdingDistance<'a, S>
where
    S: 'a,
{
    pub index: usize,
    pub solution: &'a S,
    pub rank: usize,
    pub crowding_distance: f64,
}

pub struct ObjectiveStat {
    pub spread: f64,
}

pub trait Objective {
    type Solution;

    type Distance: Sized;

    fn total_order(&self, a: &Self::Solution, b: &Self::Solution) -> Ordering;

    fn distance(&self, a: &Self::Solution, b: &Self::Solution) -> Self::Distance;
}


pub struct MultiObjective<'a, S, D>
where
    S: 'a,
    D: 'a,
{
    pub objectives: &'a [&'a dyn Objective<Solution=S, Distance=D>],
    _solution: PhantomData<S>,
    _distance: PhantomData<D>,
}

impl<'a, S, D> MultiObjective<'a, S, D>
where
    S: 'a,
    D: 'a,
{
    pub fn new(objectives: &'a [&'a dyn Objective<Solution=S, Distance=D>]) -> Self {
        Self {
            objectives,
            _solution: PhantomData,
            _distance: PhantomData,
        }
    }
}

impl<'a, S, D> DominanceOrd for MultiObjective<'a, S, D>
where
    S: 'a,
    D: 'a,
{
    type T = S;

    fn dominance_ord(&self, a: &Self::T, b: &Self::T) -> Ordering {
        let mut less_cnt = 0;
        let mut greater_cnt = 0;

        for objective in self.objectives.iter() {
            match objective.total_order(a, b) {
                Ordering::Less => {
                    less_cnt += 1;
                }
                Ordering::Greater => {
                    greater_cnt += 1;
                }
                Ordering::Equal => {}
            }
        }

        if less_cnt > 0 && greater_cnt == 0 {
            Ordering::Less
        } else if greater_cnt > 0 && less_cnt == 0 {
            Ordering::Greater
        } else {
            debug_assert!((less_cnt > 0 && greater_cnt > 0) || (less_cnt == 0 && greater_cnt == 0));
            Ordering::Equal
        }
    }
}


impl DominanceOrd for Chromosome {
    type T = Chromosome;

    fn dominance_ord(&self, a: &Self::T, b: &Self::T) -> Ordering {
        if a.makespan < b.makespan && a.mean_fidelity >= b.mean_fidelity {
            Ordering::Less
        } else if a.makespan <= b.makespan && a.mean_fidelity > b.mean_fidelity {
            Ordering::Less
        } else if a.makespan > b.makespan && a.mean_fidelity <= b.mean_fidelity {
            Ordering::Greater
        } else if a.makespan >= b.makespan && a.mean_fidelity < b.mean_fidelity {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    }
}

impl<'f, 's: 'f, S: 's> Front<'s, S> {
    pub fn next_front(self) -> Self {
        let Front {
            dominated_solutions,
            mut domination_count,
            previous_front,
            current_front,
            rank,
            solutions,
        } = self;

        // reuse the previous_front
        let mut next_front = previous_front;
        next_front.clear();

        for &p_i in current_front.iter() {
            for &q_i in dominated_solutions[p_i].iter() {
                debug_assert!(domination_count[q_i] > 0);
                domination_count[q_i] -= 1;
                if domination_count[q_i] == 0 {
                    // q_i is not dominated by any other solution. it belongs to the next front.
                    next_front.push(q_i);
                }
            }
        }

        Self {
            dominated_solutions,
            domination_count,
            previous_front: current_front,
            current_front: next_front,
            rank: rank + 1,
            solutions,
        }
    }
}

pub struct FrontElemIter<'f, 's: 'f, S: 's> {
    front: &'f Front<'s, S>,
    next_idx: SolutionIdx,
}

impl<'f, 's: 'f, S: 's> Iterator for FrontElemIter<'f, 's, S> {
    type Item = (&'s S, usize);

    fn next(&mut self) -> Option<Self::Item> {
        match self.front.current_front.get(self.next_idx) {
            Some(&solution_idx) => {
                self.next_idx += 1;
                Some((&self.front.solutions[solution_idx], solution_idx))
            }
            None => None,
        }
    }
}

impl Chromosome {
    pub fn new(config: &Config) -> Self {
        let mut rng = thread_rng();
        let order = (0..config.topological_orders.len()).choose(&mut rng).unwrap();
        let range = Uniform::new(0, config.backends);
        let random_backends = (0..config.jobs).map(|_| rng.sample(&range));
        let genes = config
            .topological_orders[order]
            .iter()
            .cloned()
            .zip(random_backends)
            .collect();

        Self { genes, order, ..Default::default() }
    }

    pub fn calculate_fitness(&mut self, config: &Config) {
        let mut acc_fidelity: f64 = 0.0;
        let mut max_makespan: u32 = 0;
        let mut job_end_times: HashMap<u32, u32> = HashMap::with_capacity(config.dependencies.len());
        let mut last_end_times: Vec<u32> = vec![0; config.backends as usize];

        for &(job, backend) in self.genes.iter() {
            let mut start_time: u32 = 0;

            if config.waiting_times[backend as usize] > start_time {
                start_time = config.waiting_times[backend as usize];
            }

            if last_end_times[backend as usize] > start_time {
                start_time = last_end_times[backend as usize];
            }

            let is_dep = config.deps_hash.contains(&job);

            if is_dep {
                for &(dep_job, dependent) in config.dependencies.iter() {
                    if dependent == job {
                        let dep_end_time = job_end_times[&dep_job];
                        if dep_end_time > start_time {
                            start_time = dep_end_time;
                        }
                    }
                }
            }

            let execution_time = config.execution_times[job as usize * config.backends as usize + backend as usize];
            let makespan = start_time + execution_time;

            last_end_times[backend as usize] = makespan;

            if is_dep {
                job_end_times.insert(job, makespan);
            }

            if makespan > max_makespan {
                max_makespan = makespan;
            }

            acc_fidelity += config.fidelities[job as usize * config.backends as usize + backend as usize];
        }

        let mean_fidelity = acc_fidelity / config.jobs as f64;

        self.makespan = max_makespan;
        self.mean_fidelity = acc_fidelity / config.jobs as f64;
        self.fitness = config.makespan_weight * (1.0 / max_makespan as f64) + config.fidelity_weight * mean_fidelity;
    }

    fn mutate(&mut self, config: &Config) {
        let mut rng = thread_rng();

        // Actually mutate the schedule by swapping backends between
        // randomly chosen jobs
        if rng.gen_bool(config.mutation_rate) {
            for _ in 0..config.mutation_pairs {
                let ind_e = rng.gen_range(0..config.jobs as usize);
                let ind_f = rng.gen_range(0..config.jobs as usize);

                let b = self.genes[ind_f].1;
                self.genes[ind_f].1 = self.genes[ind_e].1;
                self.genes[ind_e].1 = b;
            }
        }

        let mut gaps: Vec<Vec<(u32, u32)>> = vec![vec![]; config.backends as usize];
        let mut gap_sizes: Vec<u32> = vec![0; config.backends as usize];
        let mut job_end_times: HashMap<u32, u32> = HashMap::with_capacity(config.dependencies.len());
        let mut last_end_times: Vec<u32> = vec![0; config.backends as usize];
        let mut max_makespan: u32 = 0;

        // Condense the resulting schedule for better backend utilization
        for &(job, backend) in self.genes.iter() {
            let mut start_time: u32 = 0;

            if config.waiting_times[backend as usize] > start_time {
                start_time = config.waiting_times[backend as usize];
            }

            if last_end_times[backend as usize] > start_time {
                start_time = last_end_times[backend as usize];
            }

            let is_dep = config.deps_hash.contains(&job);

            if is_dep {
                for &(dep_job, dependent) in config.dependencies.iter() {
                    if dependent == job {
                        let dep_end_time = job_end_times[&dep_job];
                        if dep_end_time > start_time {
                            start_time = dep_end_time;
                        }
                    }
                }
            }

            let execution_time = config.execution_times[job as usize * config.backends as usize + backend as usize];
            let makespan = start_time + execution_time;

            let gap = start_time - last_end_times[backend as usize];
            if gap as usize > config.gap_threshold {
                gaps[backend as usize].push((last_end_times[backend as usize], start_time));
                gap_sizes[backend as usize] += gap;
            }

            last_end_times[backend as usize] = makespan;

            if is_dep {
                job_end_times.insert(job, makespan);
            }

            if makespan > max_makespan {
                max_makespan = makespan
            }
        }

        for i in 0..last_end_times.len() {
            let gap = max_makespan - last_end_times[i];
            if gap as usize > config.gap_threshold {
                gaps[i].push((last_end_times[i], max_makespan));
                gap_sizes[i] += gap;
            }
        }

        let mut makespan_indices: Vec<(u32, u32)> = last_end_times
            .iter()
            .enumerate()
            .map(|(index, &makespan)| (index as u32, makespan))
            .collect();

        makespan_indices.sort_unstable_by_key(|&(_, makespan)| std::cmp::Reverse(makespan));
        let top_n_makespan: Vec<u32> = makespan_indices
            .iter()
            .take(config.max_utilized_backends)
            .map(|&(index, _)| index)
            .collect();

        let mut idle_time_indices: Vec<(u32, u32)> = gap_sizes
            .iter()
            .enumerate()
            .map(|(index, &gap_size)| (index as u32, gap_size))
            .collect();

        idle_time_indices.sort_unstable_by_key(|&(_, total_idle_time)| std::cmp::Reverse(total_idle_time));
        let top_m_idle_time: Vec<u32> = idle_time_indices
            .iter()
            .take(config.max_underutilized_backends)
            .map(|&(index, _)| index)
            .collect();

        for i in 0..rng.gen_range(config.max_utilized_backends..(config.max_utilized_backends + config.max_underutilized_backends)) {
            let top_backend = top_n_makespan[i % top_n_makespan.len()];
            let mut top_job: u32 = 0;
            let mut top_job_idx: usize = 0;
            for j in (0..self.genes.len()).rev() {
                if self.genes[j].1 == top_backend && !config.deps_hash.contains(&self.genes[j].0) {
                    top_job = self.genes[j].0;
                    top_job_idx = j;
                    break;
                }
            }

            let idle_backend = top_m_idle_time[i % top_n_makespan.len()];
            for k in 0..gaps[idle_backend as usize].len() {
                let (end, start) = gaps[idle_backend as usize][k];
                let execution_time = config.execution_times[top_job as usize * config.backends as usize + idle_backend as usize];

                if execution_time < start - end {
                    self.genes[top_job_idx].1 = idle_backend;
                    gaps[idle_backend as usize][k].0 += execution_time;
                    break;
                }
            }
        }
    }
}

struct MakespanObjective;
struct MeanFidelityObjective;

impl Objective for MakespanObjective {
    type Solution = Chromosome;
    type Distance = f64;

    fn total_order(&self, a: &Self::Solution, b: &Self::Solution) -> Ordering {
        a.makespan.partial_cmp(&b.makespan).unwrap()
    }

    fn distance(&self, a: &Self::Solution, b: &Self::Solution) -> Self::Distance {
        (a.makespan - b.makespan) as Self::Distance
    }
}

impl Objective for MeanFidelityObjective {
    type Solution = Chromosome;
    type Distance = f64;

    fn total_order(&self, a: &Self::Solution, b: &Self::Solution) -> Ordering {
        b.mean_fidelity.partial_cmp(&a.mean_fidelity).unwrap()
    }

    fn distance(&self, a: &Self::Solution, b: &Self::Solution) -> Self::Distance {
        (b.mean_fidelity - a.mean_fidelity) as Self::Distance
    }
}


/// Perform a non-dominated sort of `solutions`. Returns the first
/// Pareto front.
pub fn non_dominated_sort<'s, S, D>(solutions: &'s [S], domination: &D) -> Front<'s, S>
where
    D: DominanceOrd<T=S>,
{
    let mut dominated_solutions: Vec<Vec<SolutionIdx>> =
        solutions.iter().map(|_| Vec::new()).collect();

    let mut domination_count: Vec<usize> = solutions.iter().map(|_| 0).collect();
    let mut current_front: Vec<SolutionIdx> = Vec::new();

    let mut iter = solutions.iter().enumerate();
    while let Some((p_i, p)) = iter.next() {
        let mut pair_iter = iter.clone();
        while let Some((q_i, q)) = pair_iter.next() {
            match domination.dominance_ord(p, q) {
                Ordering::Less => {
                    // p dominates q
                    // Add `q` to the set of solutions dominated by `p`.
                    dominated_solutions[p_i].push(q_i);
                    // q is dominated by p
                    domination_count[q_i] += 1;
                }
                Ordering::Greater => {
                    // p is dominated by q
                    // Add `p` to the set of solutions dominated by `q`.
                    dominated_solutions[q_i].push(p_i);
                    // q dominates p
                    // Increment domination counter of `p`.
                    domination_count[p_i] += 1
                }
                Ordering::Equal => {}
            }
        }
        // if domination_count drops to zero, push index to front.
        if domination_count[p_i] == 0 {
            current_front.push(p_i);
        }
    }

    Front {
        dominated_solutions,
        domination_count,
        previous_front: Vec::new(),
        current_front,
        rank: 0,
        solutions,
    }
}

pub fn assign_crowding_distance<'a, S>(
    front: &Front<'a, S>,
    multi_objective: &MultiObjective<S, f64>,
) -> Vec<AssignedCrowdingDistance<'a, S>> {
    let mut a: Vec<_> = front
        .solutions
        .iter()
        .enumerate()
        .map(|(i, s)| AssignedCrowdingDistance {
            index: i,
            solution: s,
            rank: front.rank,
            crowding_distance: 0.0,
        })
        .collect();

    multi_objective
        .objectives
        .iter()
        .for_each(|objective| {
            // First, sort according to objective
            a.sort_by(|a, b| objective.total_order(a.solution, b.solution));

            // Assign infinite crowding distance to the extremes
            {
                a.first_mut().unwrap().crowding_distance = f64::INFINITY;
                a.last_mut().unwrap().crowding_distance = f64::INFINITY;
            }

            // The distance between the "best" and "worst" solution
            // according to "objective".
            let spread = objective
                .distance(a.first().unwrap().solution, a.last().unwrap().solution)
                .abs();
            debug_assert!(spread >= 0.0);

            if spread > 0.0 {
                let norm = 1.0 / (spread * (multi_objective.objectives.len() as f64));
                debug_assert!(norm > 0.0);

                for i in 1..a.len() - 1 {
                    debug_assert!(i >= 1 && i + 1 < a.len());

                    let distance = objective
                        .distance(a[i + 1].solution, a[i - 1].solution)
                        .abs();
                    debug_assert!(distance >= 0.0);
                    a[i].crowding_distance += distance * norm;
                }
            }
        });

    a
}

fn select_and_rank<'a, S: 'a>(
    solutions: &'a [S],
    n: usize,
    multi_objective: &MultiObjective<S, f64>,
) -> Vec<AssignedCrowdingDistance<'a, S>> {
    // Cannot select more solutions than we actually have
    let n = solutions.len().min(n);
    debug_assert!(n <= solutions.len());

    let mut result = Vec::with_capacity(n);
    let mut missing_solutions = n;
    let mut front = non_dominated_sort(solutions, multi_objective);

    loop {
        let mut assigned_crowding = assign_crowding_distance(&front, multi_objective);

        if assigned_crowding.len() > missing_solutions {
            // the front does not fit in total. sort its solutions
            // according to the crowding distance and take the best
            // solutions until we have "n" solutions in the result.

            assigned_crowding.sort_by(|a, b| {
                debug_assert_eq!(a.rank, b.rank);
                a.crowding_distance
                    .partial_cmp(&b.crowding_distance)
                    .unwrap()
                    .reverse()
            });
        }

        // Take no more than `missing_solutions`
        let take = assigned_crowding.len().min(missing_solutions);

        result.extend(assigned_crowding.into_iter().take(take));

        missing_solutions -= take;
        if missing_solutions == 0 {
            break;
        }

        front = front.next_front();
    }


    debug_assert_eq!(n, result.len());

    result
}

pub fn generate_population(pop_size: usize, config: &Config) -> Vec<Chromosome> {
    (0..pop_size)
        .into_par_iter()
        .map(|_| { Chromosome::new(config) })
        .collect()
}

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

    for _ in 0..backends {
        let mean = rng.gen_range(0.0..=1.0);
        let std_dev = 1.0;

        distribution.push(Normal::new(mean, std_dev).unwrap());
    }

    for _ in 0..jobs {
        for &d in distribution.iter() {
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

fn perform_single_point_crossover(parent_1: &Chromosome, parent_2: &Chromosome, config: &Config) -> (Chromosome, Chromosome) {
    let crossover_point = thread_rng().gen_range(0..config.jobs as usize);

    let mut genes_1 = Vec::with_capacity(config.jobs as usize);
    let mut genes_2 = Vec::with_capacity(config.jobs as usize);

    genes_1.extend_from_slice(&parent_1.genes[..crossover_point]);
    genes_2.extend_from_slice(&parent_2.genes[..crossover_point]);
    genes_1.extend_from_slice(&parent_2.genes[crossover_point..]);
    genes_2.extend_from_slice(&parent_1.genes[crossover_point..]);

    (
        Chromosome { genes: genes_1, order: parent_1.order, ..Default::default() },
        Chromosome { genes: genes_2, order: parent_2.order, ..Default::default() },
    )
}

fn perform_two_point_crossover(parent_1: &Chromosome, parent_2: &Chromosome, config: &Config) -> (Chromosome, Chromosome) {
    let mut rng = thread_rng();

    let crossover_point_1 = rng.gen_range(0..config.jobs as usize);
    let crossover_point_2 = rng.gen_range(crossover_point_1..config.jobs as usize);

    let mut genes_1 = Vec::with_capacity(config.jobs as usize);
    let mut genes_2 = Vec::with_capacity(config.jobs as usize);

    genes_1.extend_from_slice(&parent_1.genes[..crossover_point_1]);
    genes_2.extend_from_slice(&parent_2.genes[..crossover_point_1]);
    genes_1.extend_from_slice(&parent_2.genes[crossover_point_1..crossover_point_2]);
    genes_2.extend_from_slice(&parent_1.genes[crossover_point_1..crossover_point_2]);
    genes_1.extend_from_slice(&parent_1.genes[crossover_point_2..]);
    genes_2.extend_from_slice(&parent_2.genes[crossover_point_2..]);

    (
        Chromosome { genes: genes_1, order: parent_1.order, ..Default::default() },
        Chromosome { genes: genes_2, order: parent_2.order, ..Default::default() },
    )
}

fn natural_selection(chromosomes: &[Chromosome], n: usize) -> Vec<Chromosome> {
    chromosomes[..std::cmp::min(n, chromosomes.len())].to_vec()
}

fn roulette_selection(chromosomes: &[Chromosome], n: usize) -> Vec<Chromosome> {
    let mut rng = thread_rng();
    let mut selected: Vec<Chromosome> = Vec::with_capacity(n);

    let sum_fitness: f64 = chromosomes.iter().map(|x| x.fitness).sum();

    for _ in 0..n {
        let u = rng.gen::<f64>() * sum_fitness;

        let mut sum = 0.0;
        for x in chromosomes.iter() {
            sum += x.fitness;
            if sum >= u {
                selected.push(x.clone());
                break;
            }
        }

        selected.push(chromosomes.last().unwrap().clone())
    }

    selected
}

fn crossover(chromosomes: &[Chromosome], config: &Config) -> Vec<Chromosome> {
    let mut groups: HashMap<usize, Vec<&Chromosome>> = HashMap::new();

    for chromosome in chromosomes.iter() {
        groups
            .entry(chromosome.order)
            .or_insert_with(Vec::new)
            .push(chromosome);
    }

    groups
        .into_par_iter() // Parallel iteration over the groups.
        .flat_map(|(_, group)| {
            group
                .par_chunks_exact(2)
                .flat_map(|pair| {
                    let (p1, p2) = (&pair[0], &pair[1]);
                    let (c1, c2) = perform_two_point_crossover(p1, p2, config);
                    vec![c1, c2]
                })
                .collect::<Vec<Chromosome>>() // Collect the offspring for this group.
        })
        .collect()
}

fn mutate(mut population: Vec<Chromosome>, config: &Config) -> Vec<Chromosome> {
    population.par_iter_mut().for_each(|chromosome| { chromosome.mutate(config) });
    population
}

fn evaluate(mut population: Vec<Chromosome>, config: &Config) -> Vec<Chromosome> {
    population.par_iter_mut().for_each(|chromosome| { chromosome.calculate_fitness(config) });
    population.sort_unstable_by(|a, b| {
        b.fitness.partial_cmp(&a.fitness).unwrap_or(Ordering::Equal)
    });

    population
}

fn evaluate_only(mut population: Vec<Chromosome>, config: &Config) -> Vec<Chromosome> {
    population.par_iter_mut().for_each(|chromosome| { chromosome.calculate_fitness(config) });
    population
}

pub fn evolve(mut population: Vec<Chromosome>, config: &Config, mut terminate: impl FnMut(&[Chromosome], i32) -> bool) -> Vec<Chromosome> {
    let mut generation = 0;

    loop {
        population = evaluate(population, config);

        let mut groups: HashSet<usize> = HashSet::new();
        for chromosome in population.iter() {
            groups.insert(chromosome.order);
        }

        if let Some(best) = population.first() {
            println!(
                "{} - Best fitness: {:.8}, makespan: {}, mean fidelity: {}, groups: {}",
                format!("Generation {:3}", generation).bold().red(),
                best.fitness,
                best.makespan,
                best.mean_fidelity,
                groups.len(),
            );
        }

        if terminate(&population, generation) {
            break;
        }

        let selected = natural_selection(&population, config.selection_size);
        let elites = population[0..config.elitism_size].to_vec();
        let children = crossover(&selected, config);
        let mutated_children = mutate(children, config);

        population = elites;
        population.extend(mutated_children);

        generation += 1;
    }

    population
}

pub fn evolve_nsga2(mut population: Vec<Chromosome>, config: &Config, mut terminate: impl FnMut(&[Chromosome], i32) -> bool) -> Vec<Chromosome> {
    let mut generation = 0;
    let mo = MultiObjective::new(&[&MakespanObjective, &MeanFidelityObjective]);

    loop {
        population = evaluate_only(population, config);

        let mut ranked_population = select_and_rank(&population, config.selection_size, &mo);
        ranked_population.sort_unstable_by(|a, b| {
            a.rank.cmp(&b.rank).then_with(|| {
                a.crowding_distance
                    .partial_cmp(&b.crowding_distance)
                    .unwrap()
                    .reverse()
            })
        });

        population = ranked_population.iter().map(|i| i.solution.clone()).collect();

        let mut groups: HashSet<usize> = HashSet::new();
        for chromosome in population.iter() {
            groups.insert(chromosome.order);
        }

        if let Some(best) = population.first() {
            println!(
                "{} - Best fitness: {:.8}, makespan: {}, mean fidelity: {}, groups: {}",
                format!("Generation {:3}", generation).bold().red(),
                best.fitness,
                best.makespan,
                best.mean_fidelity,
                groups.len(),
            );
        }

        if terminate(&population, generation) {
            break;
        }

        let selected = natural_selection(&population, config.selection_size);
        let elites = population[0..config.elitism_size].to_vec();
        let children = crossover(&selected, config);
        let mutated_children = mutate(children, config);

        population = elites;
        population.extend(mutated_children);

        generation += 1;
    }

    population
}

pub fn visualize_chromsome(chromosome: &Chromosome, config: &Config) -> Vec<(u32, u32, u32, u32)> {
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

        let execution_time = config.execution_times[job as usize * config.backends as usize + backend as usize];
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
    let max_backend = schedule.iter().map(|&(_, backend, _, _)| backend).max().unwrap_or(0);
    let max_time = schedule.iter().map(|&(_, _, _, end_time)| end_time).max().unwrap_or(0);

    // Define the chart area and margins.
    let mut chart = ChartBuilder::on(&root)
        .caption("Job Schedule", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0..max_time, 0..max_backend + 1)?;

    // Configure the chart's x and y labels.
    chart.configure_mesh()
        .x_desc("Time")
        .y_desc("Backend")
        .y_labels(max_backend as usize + 1)
        .x_labels(10)
        .draw()?;

    for (b, &t) in config.waiting_times.iter().enumerate() {
        let color = &GREY;
        chart.draw_series(vec![
            Rectangle::new(
                [(0, b as u32), (t, (b + 1) as u32)], // The rectangle spans the time and backend range.
                color.filled(),
            )
        ])?
            .label(format!("Backend {}", b))
            .legend(move |(x, y)| {
                Rectangle::new([(x, y - 5), (x + 10, y + 5)], color.filled())
            });
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

        chart.draw_series(vec![
            Rectangle::new(
                [(start_time, backend), (end_time, backend + 1)], // The rectangle spans the time and backend range.
                color.filled(),
            )
        ])?
            .label(format!("Job {}", job))
            .legend(move |(x, y)| {
                Rectangle::new([(x, y - 5), (x + 10, y + 5)], color.filled())
            });
    }

    // Configure the legend for the chart.
    chart.configure_series_labels()
        .border_style(&BLACK)
        .draw()?;

    // Save the result to the specified output path.
    root.present()?;
    println!("Chart saved to {}", output_path);
    Ok(())
}
