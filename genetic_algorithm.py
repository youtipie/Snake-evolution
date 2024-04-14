import pickle
import random
from typing import List
import numpy as np
from individual import Individual
from config import *


def init_population() -> List[Individual]:
    population = [Individual(INPUT_SIZE, HIDDEN_LAYERS_NUM, HIDDEN_LAYERS_NEURONS) for _ in range(POPULATION_SIZE)]
    return population


def fitness_eval(population: List[Individual]) -> List[float]:
    fitness_values = []
    for individual in population:
        # fitness = individual.score * 50 + individual.time_alive - (
        #         10 * len(individual.moves) / len(individual.unique_moves)
        # )
        fitness = individual.score * 500 + individual.bonus_fitness
        for move_frequency in individual.move_frequency.values():
            if move_frequency / len(individual.move_frequency.values()) > 0.33:
                fitness -= 50
        fitness_values.append(fitness)
    return fitness_values


def select_parents(population: List[Individual], fitness_values: List[int]) -> List[Individual]:
    combined = list(zip(population, fitness_values))
    sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)
    selected_parents = [sorted_combined[0][0], sorted_combined[1][0]]
    return selected_parents


def crossover(parent1: Individual, parent2: Individual) -> List[Individual]:
    population = []
    for _ in range(POPULATION_SIZE):
        offspring = Individual(
            parent1.layers[0].shape[1],
            len(parent1.layers) - 1,
            parent1.layers[-1].shape[1],
            is_copy=True
        )

        for layer1, layer2 in zip(parent1.layers, parent2.layers):
            random_crossover_probs = np.random.rand(layer1.shape[0], layer1.shape[1])
            random_crossover_probs = np.where(random_crossover_probs < CROSSOVER_RATE, 1, 0)
            mask = random_crossover_probs == 1
            layer1[mask] = layer2[mask]
            offspring.layers.append(layer1)

        for bias1, bias2 in zip(parent1.biases, parent2.biases):
            random_crossover_probs = np.random.rand(bias1.shape[0], 1)
            random_crossover_probs = np.where(random_crossover_probs < CROSSOVER_RATE, 1, 0)
            mask = random_crossover_probs == 1
            bias1[mask] = bias2[mask]
            offspring.biases.append(bias1)

        random_crossover_probs = np.random.rand(parent1.outputs.shape[0], parent1.outputs.shape[1])
        random_crossover_probs = np.where(random_crossover_probs < CROSSOVER_RATE, 1, 0)
        mask = random_crossover_probs == 1
        parent1.outputs[mask] = parent2.outputs[mask]
        offspring.outputs = parent1.outputs
        population.append(offspring)
    return population


def mutate(population: List[Individual]) -> [List[Individual], int]:
    population_copy = population[:]
    mutated_number = 0
    for i in range(len(population_copy)):
        if random.random() < MUTATION_PROB:
            mutated_number += 1
            individual = population_copy[i]
            mutated_individual = Individual(
                individual.layers[0].shape[1],
                len(individual.layers) - 1,
                individual.layers[-1].shape[1],
                is_copy=True
            )

            for layer in individual.layers:
                random_mutation_probs = np.random.rand(layer.shape[0], layer.shape[1])

                random_mutation_probs = np.where(random_mutation_probs < MUTATION_PROB,
                                                 np.random.rand() * 5 - 1, 1)
                new_layer = layer * random_mutation_probs
                mutated_individual.layers.append(new_layer)

            for bias in individual.biases:
                random_mutation_probs = np.random.rand(bias.shape[0], 1)
                random_mutation_probs = np.where(random_mutation_probs < MUTATION_PROB,
                                                 np.random.rand() * 5 - 1, 1)
                new_bias = bias * random_mutation_probs
                mutated_individual.biases.append(new_bias)

            random_mutation_probs = np.random.rand(individual.outputs.shape[0], individual.outputs.shape[1])
            random_mutation_probs = np.where(random_mutation_probs < MUTATION_PROB,
                                             np.random.rand() * 5 - 1, 1)

            new_layer = individual.outputs * random_mutation_probs
            mutated_individual.outputs = new_layer

            population_copy[i] = mutated_individual
    return population_copy, mutated_number


def save_individual(snake: Individual):
    with open("snake.obj", 'wb') as file_save:
        to_save = {"layers": snake.layers,
                   "biases": snake.biases,
                   "outputs": snake.outputs}
        pickle.dump(to_save, file_save, protocol=pickle.HIGHEST_PROTOCOL)


def load_individual(filename):
    with open(filename, 'rb') as file_save:
        snake_params = pickle.load(file_save)

        snake = Individual(INPUT_SIZE, HIDDEN_LAYERS_NUM, HIDDEN_LAYERS_NEURONS, True)
        snake.layers = snake_params["layers"]
        snake.biases = snake_params["biases"]
        snake.outputs = snake_params["outputs"]

        return snake
