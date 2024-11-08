use rand::distributions::Uniform;
use rand::prelude::*;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::cmp::Ordering;
use plotters::prelude::*;
use std::error::Error;
use plotters::style::full_palette::GREY;

#[derive(Clone, Debug, Default)]
pub struct Config {
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

pub trait Chromosome: Sized + Send + Sync + Clone + std::fmt::Debug {
    fn new(config: &Config) -> Self;
    fn calculate_fitness(&mut self, config: &Config);
    fn mutate(&mut self, config: &Config);
    fn fitness(&self) -> f64;
    fn order(&self) -> usize;
}

#[derive(Clone, Debug, Default)]
pub struct ScheduleChromosome {
    pub genes: Vec<(u32, u32)>,
    pub order: usize,
    pub fitness: f64,
    pub makespan: u32,
    pub mean_fidelity: f64,
}

impl Chromosome for ScheduleChromosome {
    fn new(config: &Config) -> Self {
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

    fn calculate_fitness(&mut self, config: &Config) {
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

    fn fitness(&self) -> f64 {
        self.fitness
    }

    fn order(&self) -> usize {
        self.order
    }
}

pub fn generate_population<C: Chromosome>(config: &Config) -> Vec<C> {
    (0..config.population_size)
        .into_par_iter()
        .map(|_| { Chromosome::new(config) })
        .collect()
}

fn perform_single_point_crossover(parent_1: &ScheduleChromosome, parent_2: &ScheduleChromosome, config: &Config) -> (ScheduleChromosome, ScheduleChromosome) {
    let crossover_point = thread_rng().gen_range(0..config.jobs as usize);

    let mut genes_1 = Vec::with_capacity(config.jobs as usize);
    let mut genes_2 = Vec::with_capacity(config.jobs as usize);

    genes_1.extend_from_slice(&parent_1.genes[..crossover_point]);
    genes_2.extend_from_slice(&parent_2.genes[..crossover_point]);
    genes_1.extend_from_slice(&parent_2.genes[crossover_point..]);
    genes_2.extend_from_slice(&parent_1.genes[crossover_point..]);

    (
        ScheduleChromosome { genes: genes_1, order: parent_1.order, ..Default::default() },
        ScheduleChromosome { genes: genes_2, order: parent_2.order, ..Default::default() },
    )
}

fn perform_two_point_crossover(parent_1: &ScheduleChromosome, parent_2: &ScheduleChromosome, config: &Config) -> (ScheduleChromosome, ScheduleChromosome) {
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
        ScheduleChromosome { genes: genes_1, order: parent_1.order, ..Default::default() },
        ScheduleChromosome { genes: genes_2, order: parent_2.order, ..Default::default() },
    )
}

pub fn natural_selection<C: Chromosome>(chromosomes: &[C], n: usize) -> &[C] {
    &chromosomes[..std::cmp::min(n, chromosomes.len())]
}

pub fn roulette_selection<C: Chromosome>(chromosomes: &[C], n: usize) -> Vec<C> {
    let mut rng = thread_rng();
    let mut selected: Vec<C> = Vec::with_capacity(n);

    let sum_fitness: f64 = chromosomes.iter().map(|x| x.fitness()).sum();

    for _ in 0..n {
        let u = rng.gen::<f64>() * sum_fitness;

        let mut sum = 0.0;
        for x in chromosomes.iter() {
            sum += x.fitness();
            if sum >= u {
                selected.push(x.clone());
                break;
            }
        }

        selected.push(chromosomes.last().unwrap().clone())
    }

    selected
}

pub fn crossover<C: Chromosome>(chromosomes: &[C], config: &Config) -> Vec<C> {
    let mut groups: HashMap<usize, Vec<C>> = HashMap::new();

    for chromosome in chromosomes.iter() {
        groups
            .entry(chromosome.order())
            .or_insert_with(Vec::new)
            .push(chromosome.clone());
    }

    groups
        .into_par_iter()
        .flat_map(|(_, group)| {
            group
                .par_chunks_exact(2)
                .flat_map(|pair| {
                    let (p1, p2) = (&pair[0], &pair[1]);
                    let (c1, c2) = perform_two_point_crossover(p1, p2, config);
                    vec![c1, c2]
                })
                .collect::<Vec<ScheduleChromosome>>() // Collect the offspring for this group.
        })
        .collect()
}

pub fn mutate<C: Chromosome>(mut population: Vec<C>, config: &Config) -> Vec<C> {
    population.par_iter_mut().for_each(|chromosome| { chromosome.mutate(config) });
    population
}

pub fn evaluate<C: Chromosome>(mut population: Vec<C>, config: &Config) -> Vec<C> {
    population.par_iter_mut().for_each(|chromosome| { chromosome.calculate_fitness(config) });
    population.sort_unstable_by(|a, b| {
        b.fitness().partial_cmp(&a.fitness()).unwrap_or(Ordering::Equal)
    });

    population
}

pub fn evolve<C: Chromosome>(mut population: Vec<C>, config: &Config, mut terminate: impl FnMut(&[C], i32) -> bool) -> Vec<C> {
    let mut generation = 0;

    loop {
        population = evaluate(population, config);

        if terminate(&population, generation) {
            break;
        }

        let selected = natural_selection(&population, config.selection_size);
        let elites = population[0..config.elitism_size].to_vec();
        let children = crossover(selected, config);
        let mutated_children = mutate(children, config);

        population = elites;
        population.extend(mutated_children);

        generation += 1;
    }

    population
}

pub fn visualize_chromsome(chromosome: &ScheduleChromosome, config: &Config) -> Vec<(u32, u32, u32, u32)> {
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
