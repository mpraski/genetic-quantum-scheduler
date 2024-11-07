use std::cmp::Ordering;
use crate::dominance_ord::DominanceOrd;
use crate::nsga2::MultiObjective;

type SolutionIdx = usize;

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
    pub solution: &'a S,
    pub rank: usize,
    pub crowding_distance: f64,
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