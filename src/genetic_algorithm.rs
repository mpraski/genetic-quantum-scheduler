use std::fmt::Debug;

// This trait represents a chromosome - a single solution to
// the problem we're solving
pub trait Chromosome: Send + Sync + Debug + Clone {
    fn fitness(&self) -> f64;
    fn order(&self) -> usize;
}

// This trait represents a configuration of the algorithm
pub trait Meta: Send + Sync + Debug + Clone {
    fn selection_size(&self) -> usize;
}

// This trait represents the stopping condition of the algorithm
pub trait Evaluator<C: Chromosome>: Send + Sync + Debug {
    fn can_terminate(&mut self, chromosomes: &[C], generation: i32) -> bool;
}

// This trait encapsulates the optimizer logic
pub trait Optimizer<C: Chromosome>: Send + Sync + Debug {
    fn optimize(&mut self, eval: Box<dyn Evaluator<C>>) -> Vec<C>;
}

// This trait encapsulates the underlying genetic (or a variation thereof)
// algorithm used by the optimizer to find the solution
pub trait Algorithm<M: Meta, C: Chromosome>: Send + Sync + Debug {
    fn meta(&self) -> &M;
    fn generate(&self) -> Vec<C>;
    fn evaluate(&self, population: Vec<C>) -> Vec<C>;
    fn evaluate_only(&self, population: Vec<C>) -> Vec<C>;
    fn mutate(&self, population: Vec<C>) -> Vec<C>;
    fn crossover(&self, parents: &Vec<C>) -> Vec<C>;
    fn select(&self, chromosomes: &[C]) -> Vec<C>;
    fn elitism(&self, population: &[C]) -> Vec<C>;
}
