use std::cmp::Ordering;
use std::collections::HashSet;
use std::marker::PhantomData;
use colored::Colorize;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::prelude::*;
use crate::fast_nondominating_sort::{assign_crowding_distance, non_dominated_sort, AssignedCrowdingDistance};
use crate::genetic_algo::{Chromosome, Config};

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

fn evaluate_only(mut population: Vec<Chromosome>, config: &Config) -> Vec<Chromosome> {
    population.par_iter_mut().for_each(|chromosome| { chromosome.calculate_fitness(config) });
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

        let selected = crate::genetic_algo::natural_selection(&population, config.selection_size);
        let elites = population[0..config.elitism_size].to_vec();
        let children = crate::genetic_algo::crossover(&selected, config);
        let mutated_children = crate::genetic_algo::mutate(children, config);

        population = elites;
        population.extend(mutated_children);

        generation += 1;
    }

    population
}