use std::fmt::Debug;

pub trait Chromosome: Send + Sync + Debug + Clone {
    fn fitness(&self) -> f64;
    fn order(&self) -> usize;
}

pub trait Meta: Send + Sync + Debug + Clone {
    fn selection_size(&self) -> usize;
}

pub trait Evaluator<C: Chromosome>: Send + Sync + Debug {
    fn can_terminate(&mut self, chromosomes: &[C], generation: i32) -> bool;
}

pub trait Optimizer<C: Chromosome>: Send + Sync + Debug {
    fn optimize(&mut self, eval: Box<dyn Evaluator<C>>) -> Vec<C>;
}

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
