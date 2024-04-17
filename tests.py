import unittest
from genetic_algorithm import *


class TestGeneticAlgorithm(unittest.TestCase):
    def test_fitness_eval(self):
        population = init_population()
        for ind in population:
            ind.score = random.randint(0, 50)

        fitness_values = fitness_eval(population)
        self.assertEqual(len(fitness_values), POPULATION_SIZE)
        self.assertTrue(isinstance(fitness_values, list))
        self.assertIsNotNone(fitness_values)

        individual = Individual(INPUT_SIZE, HIDDEN_LAYERS_NUM, HIDDEN_LAYERS_NEURONS)
        individual.score = 5
        individual.bonus_fitness = 10
        individual.time_alive = 100
        fitness = fitness_eval([individual])[0]
        self.assertIsNotNone(fitness)
        self.assertEqual(fitness, 75)

    def test_generate_child(self):
        parents = init_population()[:100]
        children, mutated_number = generate_child(parents)
        self.assertEqual(len(children), int(POPULATION_SIZE * 0.1))
        self.assertIsInstance(children[0], Individual)
        for layer1, layer2 in zip(children[0].layers, parents[0].layers):
            self.assertEqual(layer1.shape, layer2.shape)
        for bias1, bias1 in zip(children[0].biases, parents[0].biases):
            self.assertEqual(bias1.shape, bias1.shape)
        self.assertEqual(children[0].outputs.shape, parents[0].outputs.shape)

        self.assertIsNotNone(mutated_number)
        self.assertIsInstance(mutated_number, int)
        self.assertTrue(0 <= mutated_number < int(POPULATION_SIZE * 0.1))

    def test_select_new_population(self):
        population = init_population()
        for ind in population:
            ind.score = random.randint(0, 50)

        population[0].score = 100
        fitness_scores = fitness_eval(population)
        children, mutated_number, best_individual = select_new_population(population, fitness_scores)

        self.assertIsNotNone(children)
        self.assertEqual(len(children), POPULATION_SIZE)
        self.assertIsInstance(children[0], Individual)

        self.assertIsNotNone(mutated_number)
        self.assertIsInstance(mutated_number, int)
        self.assertTrue(0 <= mutated_number < POPULATION_SIZE)

        self.assertIsNotNone(best_individual)
        self.assertIsInstance(best_individual, Individual)
        self.assertTrue(best_individual.score != 0)
        self.assertTrue(best_individual.score == 100)
        self.assertTrue(best_individual == population[0])

    def test_single_point_crossover(self):
        parent1 = Individual(INPUT_SIZE, HIDDEN_LAYERS_NUM, HIDDEN_LAYERS_NEURONS)
        parent2 = Individual(INPUT_SIZE, HIDDEN_LAYERS_NUM, HIDDEN_LAYERS_NEURONS)
        offspring = single_point_crossover(parent1, parent2)
        self.assertIsInstance(offspring, Individual)
        self.assertIsNotNone(offspring)

        for parent in [parent1, parent2]:
            for layer1, layer2 in zip(offspring.layers, parent.layers):
                self.assertEqual(layer1.shape, layer2.shape)
                self.assertTrue((layer1 != layer2).any())
            for bias1, bias2 in zip(offspring.biases, parent.biases):
                self.assertEqual(bias1.shape, bias2.shape)
                self.assertTrue((bias1 != bias2).any())
            self.assertEqual(offspring.outputs.shape, parent.outputs.shape)
            self.assertTrue((offspring.outputs == parent.outputs).any())

    def test_uniform_crossover(self):
        parent1 = Individual(INPUT_SIZE, HIDDEN_LAYERS_NUM, HIDDEN_LAYERS_NEURONS)
        parent2 = Individual(INPUT_SIZE, HIDDEN_LAYERS_NUM, HIDDEN_LAYERS_NEURONS)
        offspring = uniform_crossover(parent1, parent2)
        self.assertIsInstance(offspring, Individual)
        self.assertIsNotNone(offspring)

        for parent in [parent1, parent2]:
            for layer1, layer2 in zip(offspring.layers, parent.layers):
                self.assertEqual(layer1.shape, layer2.shape)
                self.assertTrue((layer1 != layer2).any())
            for bias1, bias2 in zip(offspring.biases, parent.biases):
                self.assertEqual(bias1.shape, bias2.shape)
                self.assertTrue((bias1 != bias2).any())
            self.assertEqual(offspring.outputs.shape, parent.outputs.shape)
            self.assertTrue((offspring.outputs == parent.outputs).any())

    def test_mutate(self):
        individual = Individual(INPUT_SIZE, HIDDEN_LAYERS_NUM, HIDDEN_LAYERS_NEURONS)
        mutated_individual = mutate(individual)
        self.assertIsInstance(mutated_individual, Individual)
        self.assertIsNotNone(mutated_individual)

        for layer1, layer2 in zip(mutated_individual.layers, individual.layers):
            self.assertEqual(layer1.shape, layer2.shape)
            self.assertTrue((layer1 != layer2).any())
        for bias1, bias2 in zip(mutated_individual.biases, individual.biases):
            self.assertEqual(bias1.shape, bias2.shape)
            self.assertTrue((bias1 != bias2).any())
        self.assertEqual(mutated_individual.outputs.shape, individual.outputs.shape)
        self.assertTrue((mutated_individual.outputs == individual.outputs).any())

    if __name__ == '__main__':
        unittest.main()
