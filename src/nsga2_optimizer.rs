use crate::genetic_algorithm::{Algorithm, Evaluator, Optimizer};
use crate::genetic_optimizer::{QuantumSchedule, SchedulingConfig};
use crate::non_dominated_sort::{
    assign_crowding_distance, non_dominated_sort, AssignedCrowdingDistance,
};
use std::cmp::Ordering;
use std::marker::PhantomData;

#[derive(Debug)]
pub struct NSGA2Optimizer {
    pub algorithm: Box<dyn Algorithm<SchedulingConfig, QuantumSchedule>>,
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
    pub objectives: &'a [&'a dyn Objective<Solution = S, Distance = D>],
    _solution: PhantomData<S>,
    _distance: PhantomData<D>,
}

impl<'a, S, D> MultiObjective<'a, S, D>
where
    S: 'a,
    D: 'a,
{
    pub fn new(objectives: &'a [&'a dyn Objective<Solution = S, Distance = D>]) -> Self {
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
    type Solution = QuantumSchedule;
    type Distance = f64;

    fn total_order(&self, a: &Self::Solution, b: &Self::Solution) -> Ordering {
        a.makespan.partial_cmp(&b.makespan).unwrap()
    }

    fn distance(&self, a: &Self::Solution, b: &Self::Solution) -> Self::Distance {
        (a.makespan - b.makespan) as Self::Distance
    }
}

impl Objective for MeanFidelityObjective {
    type Solution = QuantumSchedule;
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

impl Optimizer<QuantumSchedule> for NSGA2Optimizer {
    fn optimize(&mut self, eval: &mut Box<dyn Evaluator<QuantumSchedule>>) -> Vec<QuantumSchedule> {
        let mut generation = 0;
        let mut population: Vec<QuantumSchedule> = self.algorithm.generate();
        let mo = MultiObjective::new(&[&MakespanObjective, &MeanFidelityObjective]);

        loop {
            population = self.algorithm.evaluate_only(population);

            let mut ranked_population =
                select_and_rank(&population, self.algorithm.meta().selection_size, &mo);
            ranked_population.sort_unstable_by(|a, b| {
                a.rank.cmp(&b.rank).then_with(|| {
                    a.crowding_distance
                        .partial_cmp(&b.crowding_distance)
                        .unwrap()
                        .reverse()
                })
            });

            population = ranked_population
                .iter()
                .map(|i| i.solution.clone())
                .collect();

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
