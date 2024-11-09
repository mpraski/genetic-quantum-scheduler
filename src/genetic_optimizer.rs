use crate::dominance_ord::DominanceOrd;
use crate::genetic_algorithm::{Algorithm, Chromosome, Evaluator, Meta, Optimizer};
use colored::Colorize;
use rand::distributions::Uniform;
use rand::prelude::IteratorRandom;
use rand::prelude::*;
use rand::thread_rng;
use rayon::iter::{IntoParallelIterator, IntoParallelRefMutIterator};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;

#[derive(Clone, Debug, Default)]
pub struct SchedulingConfig {
    pub nsga2: bool,
    pub jobs: u32,
    pub backends: u32,
    pub deps_hash: HashSet<u32>,
    pub fidelities: Vec<f64>,
    pub dependencies: Vec<(u32, u32)>,
    pub waiting_times: Vec<u32>,
    pub execution_times: Vec<u32>,
    pub topological_orders: Vec<Vec<u32>>,
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
}

impl Meta for SchedulingConfig {
    fn selection_size(&self) -> usize {
        self.selection_size
    }
}

#[derive(Debug)]
pub struct SchedulingAlgorithm {
    pub config: SchedulingConfig,
}

#[derive(Debug)]
pub struct GeneticOptimizer {
    pub algorithm: Box<dyn Algorithm<SchedulingConfig, QuantumSchedule>>,
}

#[derive(Debug, Default)]
pub struct SchedulingEvaluator {
    pub best_fitness: f64,
    pub best_fitness_count: u32,
    pub generations: i32,
}

impl Evaluator<QuantumSchedule> for SchedulingEvaluator {
    fn can_terminate(&mut self, chromosomes: &[QuantumSchedule], generation: i32) -> bool {
        let mut groups: HashSet<usize> = HashSet::new();
        for chromosome in chromosomes.iter() {
            groups.insert(chromosome.order);
        }

        if let Some(best) = chromosomes.first() {
            println!(
                "{} - Best fitness: {:.8}, makespan: {}, mean fidelity: {}, groups: {}",
                format!("Generation {:3}", generation).bold().red(),
                best.fitness,
                best.makespan,
                best.mean_fidelity,
                groups.len(),
            );

            if self.best_fitness < best.fitness() {
                self.best_fitness = best.fitness();
                self.best_fitness_count = 0;
            } else {
                self.best_fitness_count += 1;
            }
        }

        self.generations = generation;

        generation == 200 || self.best_fitness_count == 20
    }
}
#[derive(Clone, Debug, Default)]
pub struct QuantumSchedule {
    pub genes: Vec<(u32, u32)>,
    pub order: usize,
    pub fitness: f64,
    pub makespan: u32,
    pub mean_fidelity: f64,
}

impl Algorithm<SchedulingConfig, QuantumSchedule> for SchedulingAlgorithm {
    fn meta(&self) -> &SchedulingConfig {
        &self.config
    }

    fn generate(&self) -> Vec<QuantumSchedule> {
        (0..self.config.population_size)
            .into_par_iter()
            .map(|_| {
                let mut rng = thread_rng();
                let order = (0..self.config.topological_orders.len())
                    .choose(&mut rng)
                    .unwrap();
                let range = Uniform::new(0, self.config.backends);
                let random_backends = (0..self.config.jobs).map(|_| rng.sample(&range));
                let genes = self.config.topological_orders[order]
                    .iter()
                    .cloned()
                    .zip(random_backends)
                    .collect();

                QuantumSchedule {
                    genes,
                    order,
                    ..Default::default()
                }
            })
            .collect()
    }

    fn evaluate(&self, mut population: Vec<QuantumSchedule>) -> Vec<QuantumSchedule> {
        population = self.evaluate_only(population);

        population.sort_unstable_by(|a, b| {
            b.fitness()
                .partial_cmp(&a.fitness())
                .unwrap_or(Ordering::Equal)
        });

        population
    }

    fn evaluate_only(&self, mut population: Vec<QuantumSchedule>) -> Vec<QuantumSchedule> {
        population.par_iter_mut().for_each(|chromosome| {
            let mut acc_fidelity: f64 = 0.0;
            let mut max_makespan: u32 = 0;
            let mut job_end_times: HashMap<u32, u32> =
                HashMap::with_capacity(self.config.dependencies.len());
            let mut last_end_times: Vec<u32> = vec![0; self.config.backends as usize];

            for &(job, backend) in chromosome.genes.iter() {
                let mut start_time: u32 = 0;

                if self.config.waiting_times[backend as usize] > start_time {
                    start_time = self.config.waiting_times[backend as usize];
                }

                if last_end_times[backend as usize] > start_time {
                    start_time = last_end_times[backend as usize];
                }

                let is_dep = self.config.deps_hash.contains(&job);

                if is_dep {
                    for &(dep_job, dependent) in self.config.dependencies.iter() {
                        if dependent == job {
                            let dep_end_time = job_end_times[&dep_job];
                            if dep_end_time > start_time {
                                start_time = dep_end_time;
                            }
                        }
                    }
                }

                let execution_time = self.config.execution_times
                    [job as usize * self.config.backends as usize + backend as usize];
                let makespan = start_time + execution_time;

                last_end_times[backend as usize] = makespan;

                if is_dep {
                    job_end_times.insert(job, makespan);
                }

                if makespan > max_makespan {
                    max_makespan = makespan;
                }

                acc_fidelity += self.config.fidelities
                    [job as usize * self.config.backends as usize + backend as usize];
            }

            let mean_fidelity = acc_fidelity / self.config.jobs as f64;

            chromosome.makespan = max_makespan;
            chromosome.mean_fidelity = acc_fidelity / self.config.jobs as f64;
            chromosome.fitness = self.config.makespan_weight * (1.0 / max_makespan as f64)
                + self.config.fidelity_weight * mean_fidelity;
        });

        population
    }

    fn mutate(&self, mut population: Vec<QuantumSchedule>) -> Vec<QuantumSchedule> {
        population.par_iter_mut().for_each(|chromosome| {
            let mut rng = thread_rng();

            // Actually mutate the schedule by swapping backends between
            // randomly chosen jobs
            if rng.gen_bool(self.config.mutation_rate) {
                for _ in 0..self.config.mutation_pairs {
                    let ind_e = rng.gen_range(0..self.config.jobs as usize);
                    let ind_f = rng.gen_range(0..self.config.jobs as usize);

                    let b = chromosome.genes[ind_f].1;
                    chromosome.genes[ind_f].1 = chromosome.genes[ind_e].1;
                    chromosome.genes[ind_e].1 = b;
                }
            }

            let mut gaps: Vec<Vec<(u32, u32)>> = vec![vec![]; self.config.backends as usize];
            let mut gap_sizes: Vec<u32> = vec![0; self.config.backends as usize];
            let mut job_end_times: HashMap<u32, u32> =
                HashMap::with_capacity(self.config.dependencies.len());
            let mut last_end_times: Vec<u32> = vec![0; self.config.backends as usize];
            let mut max_makespan: u32 = 0;

            // Condense the resulting schedule for better backend utilization
            for &(job, backend) in chromosome.genes.iter() {
                let mut start_time: u32 = 0;

                if self.config.waiting_times[backend as usize] > start_time {
                    start_time = self.config.waiting_times[backend as usize];
                }

                if last_end_times[backend as usize] > start_time {
                    start_time = last_end_times[backend as usize];
                }

                let is_dep = self.config.deps_hash.contains(&job);

                if is_dep {
                    for &(dep_job, dependent) in self.config.dependencies.iter() {
                        if dependent == job {
                            let dep_end_time = job_end_times[&dep_job];
                            if dep_end_time > start_time {
                                start_time = dep_end_time;
                            }
                        }
                    }
                }

                let execution_time = self.config.execution_times
                    [job as usize * self.config.backends as usize + backend as usize];
                let makespan = start_time + execution_time;

                let gap = start_time - last_end_times[backend as usize];
                if gap as usize > self.config.gap_threshold {
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
                if gap as usize > self.config.gap_threshold {
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
                .take(self.config.max_utilized_backends)
                .map(|&(index, _)| index)
                .collect();

            let mut idle_time_indices: Vec<(u32, u32)> = gap_sizes
                .iter()
                .enumerate()
                .map(|(index, &gap_size)| (index as u32, gap_size))
                .collect();

            idle_time_indices
                .sort_unstable_by_key(|&(_, total_idle_time)| std::cmp::Reverse(total_idle_time));
            let top_m_idle_time: Vec<u32> = idle_time_indices
                .iter()
                .take(self.config.max_underutilized_backends)
                .map(|&(index, _)| index)
                .collect();

            for i in 0..rng.gen_range(
                self.config.max_utilized_backends
                    ..(self.config.max_utilized_backends + self.config.max_underutilized_backends),
            ) {
                let top_backend = top_n_makespan[i % top_n_makespan.len()];
                let mut top_job: u32 = 0;
                let mut top_job_idx: usize = 0;
                for j in (0..chromosome.genes.len()).rev() {
                    if chromosome.genes[j].1 == top_backend
                        && !self.config.deps_hash.contains(&chromosome.genes[j].0)
                    {
                        top_job = chromosome.genes[j].0;
                        top_job_idx = j;
                        break;
                    }
                }

                let idle_backend = top_m_idle_time[i % top_n_makespan.len()];
                for k in 0..gaps[idle_backend as usize].len() {
                    let (end, start) = gaps[idle_backend as usize][k];
                    let execution_time = self.config.execution_times
                        [top_job as usize * self.config.backends as usize + idle_backend as usize];

                    if execution_time < start - end {
                        chromosome.genes[top_job_idx].1 = idle_backend;
                        gaps[idle_backend as usize][k].0 += execution_time;
                        break;
                    }
                }
            }
        });

        population
    }

    fn crossover(&self, parents: &Vec<QuantumSchedule>) -> Vec<QuantumSchedule> {
        let mut groups: HashMap<usize, Vec<&QuantumSchedule>> = HashMap::new();

        for chromosome in parents {
            groups
                .entry(chromosome.order())
                .or_insert_with(Vec::new)
                .push(chromosome);
        }

        groups
            .into_par_iter()
            .flat_map(|(_, group)| {
                group
                    .par_chunks_exact(2)
                    .flat_map(|pair| {
                        let (p1, p2) = (&pair[0], &pair[1]);
                        let (c1, c2) = perform_two_point_crossover(p1, p2, &self.config);
                        vec![c1, c2]
                    })
                    .collect::<Vec<QuantumSchedule>>() // Collect the offspring for this group.
            })
            .collect()
    }

    fn select(&self, chromosomes: &[QuantumSchedule]) -> Vec<QuantumSchedule> {
        chromosomes[..std::cmp::min(self.config.selection_size, chromosomes.len())].to_vec()
    }

    fn elitism(&self, population: &[QuantumSchedule]) -> Vec<QuantumSchedule> {
        population[..std::cmp::min(self.config.elitism_size, population.len())].to_vec()
    }
}

impl Chromosome for QuantumSchedule {
    fn fitness(&self) -> f64 {
        self.fitness
    }

    fn order(&self) -> usize {
        self.order
    }
}

impl DominanceOrd for QuantumSchedule {
    type T = QuantumSchedule;

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

impl Optimizer<QuantumSchedule> for GeneticOptimizer {
    fn optimize(&mut self, mut eval: Box<dyn Evaluator<QuantumSchedule>>) -> Vec<QuantumSchedule> {
        let mut generation = 0;
        let mut population: Vec<QuantumSchedule> = self.algorithm.generate();

        loop {
            population = self.algorithm.evaluate(population);

            if eval.can_terminate(&population, generation) {
                break;
            }

            let selected = self.algorithm.select(&population);
            let elites = self.algorithm.elitism(&population);
            let children = self.algorithm.crossover(&selected);
            let mutated_children = self.algorithm.mutate(children);

            population = elites;
            population.extend(mutated_children);

            generation += 1;
        }

        population
    }
}

fn perform_single_point_crossover(
    parent_1: &QuantumSchedule,
    parent_2: &QuantumSchedule,
    config: &SchedulingConfig,
) -> (QuantumSchedule, QuantumSchedule) {
    let crossover_point = thread_rng().gen_range(0..config.jobs as usize);

    let mut genes_1 = Vec::with_capacity(config.jobs as usize);
    let mut genes_2 = Vec::with_capacity(config.jobs as usize);

    genes_1.extend_from_slice(&parent_1.genes[..crossover_point]);
    genes_2.extend_from_slice(&parent_2.genes[..crossover_point]);
    genes_1.extend_from_slice(&parent_2.genes[crossover_point..]);
    genes_2.extend_from_slice(&parent_1.genes[crossover_point..]);

    (
        QuantumSchedule {
            genes: genes_1,
            order: parent_1.order,
            ..Default::default()
        },
        QuantumSchedule {
            genes: genes_2,
            order: parent_2.order,
            ..Default::default()
        },
    )
}

fn perform_two_point_crossover(
    parent_1: &QuantumSchedule,
    parent_2: &QuantumSchedule,
    config: &SchedulingConfig,
) -> (QuantumSchedule, QuantumSchedule) {
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
        QuantumSchedule {
            genes: genes_1,
            order: parent_1.order,
            ..Default::default()
        },
        QuantumSchedule {
            genes: genes_2,
            order: parent_2.order,
            ..Default::default()
        },
    )
}
