# Genetic Quantum Scheduler

This repo defines the genetic algorithm optimizers for quantum job scheduling.

This project implements a Genetic Algorithm (GA) with support for multi-objective optimization using the NSGA-II (Non-dominated Sorting Genetic Algorithm II) algorithm. The primary purpose of this codebase is to demonstrate and execute optimization tasks on a population of solutions, optimizing for multiple objectives like minimizing makespan and maximizing fidelity in scheduling contexts.

The key algorithms implemented include:

- Genetic Algorithm: Implements core operations like mutation, crossover, and selection. 
- NSGA-II Algorithm: Adds support for multi-objective optimization, leveraging dominance-based selection to maintain diversity in solutions. 

The repository structure provides modules for each step of the algorithm, encapsulating tasks like data generation, optimization, visualization, and dominance operations.


## File Structure and Overview

1. **`main.rs`**
    - **Purpose**: The main entry point of the application.
    - **Description**: This file initializes the configuration for the genetic algorithm, creates an optimizer, runs the optimization process, and handles output visualization.
    - **Core Functionality**:
        - Sets up configuration parameters.
        - Calls the appropriate optimizer (e.g., NSGA-II).
        - Displays the results using the visualization module.

2. **`demo_data.rs`**
    - **Purpose**: Generates demo data for testing the genetic algorithm.
    - **Description**: Contains functions to generate random scheduling data and sets up initial configurations for genes, chromosomes, and evaluators.
    - **Core Functionality**:
        - Creates test data, such as random chromosomes and configurations.
        - Supports benchmarking with adjustable parameters for various test runs.

3. **`genetic_algorithm.rs`**
    - **Purpose**: Defines the core Genetic Algorithm (GA) logic.
    - **Description**: Implements standard GA operations, including mutation, crossover, selection, and evaluation of fitness. Provides a flexible interface for adding different genetic operators.
    - **Core Functionality**:
        - **Mutation**: Alters genes of chromosomes to introduce variation.
        - **Crossover**: Combines genes from parent chromosomes to produce offspring.
        - **Selection**: Selects the fittest chromosomes based on fitness scores.
        - **Fitness Evaluation**: Calculates fitness for chromosomes.

4. **`genetic_optimizer.rs`**
    - **Purpose**: Provides an optimization interface for the genetic algorithm.
    - **Description**: Defines a `GeneticOptimizer` struct which initializes a genetic algorithm and iterates through generations of evolution to optimize a population.
    - **Core Functionality**:
        - Executes GA operations across generations.
        - Manages population size, mutation rates, and other configurable parameters.

5. **`nsga2_optimizer.rs`**
    - **Purpose**: Implements the NSGA-II algorithm for multi-objective optimization.
    - **Description**: Extends the `GeneticOptimizer` with non-dominated sorting and Pareto-based selection for handling multiple objectives.
    - **Core Functionality**:
        - **Non-dominated Sorting**: Categorizes solutions into Pareto fronts.
        - **Crowding Distance Calculation**: Maintains diversity within solutions on the same front.
        - **Multi-objective Selection**: Selects solutions based on Pareto dominance rather than single fitness values.

6. **`dominance_ord.rs`**
    - **Purpose**: Defines dominance ordering for solutions.
    - **Description**: Implements functions to compare chromosomes based on their dominance relationships (i.e., whether one solution dominates another).
    - **Core Functionality**:
        - Provides logic for checking whether one chromosome dominates another.
        - Useful in multi-objective optimization where a solution may be preferable based on multiple criteria.

7. **`non_dominated_sort.rs`**
    - **Purpose**: Implements non-dominated sorting, a key part of NSGA-II.
    - **Description**: Provides sorting functions that group chromosomes by Pareto fronts, necessary for NSGA-II.
    - **Core Functionality**:
        - Sorts chromosomes into non-dominated levels (fronts).
        - Allows selection from each front to preserve solution diversity.

8. **`visualization.rs`**
    - **Purpose**: Handles the visualization of algorithm results.
    - **Description**: Provides utilities for visualizing the output of the genetic algorithm, particularly useful for inspecting Pareto fronts in multi-objective problems.
    - **Core Functionality**:
        - Plots results of the optimization (e.g., solutions on the Pareto front).
        - Allows visual inspection of solution diversity and effectiveness.

---

## How to Run the Code

1. **Setup**:
    - Ensure you have Rust installed. If not, install it from [rust-lang.org](https://www.rust-lang.org/).
    - Clone the repository and navigate to the project directory.

2. **Running the Main Application**:
   ```bash
   cargo run --release
   ```