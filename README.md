# Optimization Algorithms

This branch contains two optimization algorithms: **Genetic Algorithm (GA)** and **Imperialist Competitive Algorithm (ICA)**. Both algorithms are implemented using functions from the **NumPy** library.

The objective function in both algorithms is designed to be minimized.

## 1. Genetic Algorithm (GA)

The Genetic Algorithm works by creating an initial population of genes and combining the best-performing genes to generate offspring.

The generated offspring are then evaluated based on how effectively they minimize the objective function. The lower the objective function value, the higher the fitness score.

The offspring with higher fitness scores are selected and added to the population. This process continues iteratively until the objective function reaches its minimum value, ideally leaving only one best-performing gene.

## 2. Imperialist Competitive Algorithm (ICA)

The Imperialist Competitive Algorithm starts by generating a number of countries. Each country is assigned a cost based on the objective function.

Based on their costs, the countries are divided into two groups:

- **Imperialists:** Countries with better objective function values.
- **Colonies:** Countries with less favorable objective function values.

The colonies gradually move toward their respective imperialists. If a colony becomes stronger than its imperialist and successfully takes over its position, it becomes an imperialist itself.

This process continues through imperialistic competition until the objective function reaches its minimum value, ideally leaving only one imperialist.
