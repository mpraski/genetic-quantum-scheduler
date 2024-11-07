use std::cmp::Ordering;
use crate::genetic_algo::{Chromosome};
use crate::nsga2::MultiObjective;

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
