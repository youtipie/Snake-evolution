import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import math
import os.path
import pandas as pd
import pygame
import matplotlib.pyplot as plt
from time import time, sleep

from statistics import mean
from genetic_algorithm import *

pygame.font.init()
win_width = 600
win_height = 600
stat_font = pygame.font.SysFont("comicsans", 50)
cellSize = 30

showcase = False


class Snake:
    color = (46, 142, 212)

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.dx = cellSize
        self.dy = 0
        self.body = [
            [self.x, self.y],
            [self.x - cellSize, self.y],
            [self.x - cellSize * 2, self.y]
        ]

    def move(self):
        self.x += self.dx
        self.y += self.dy
        self.body.insert(0, [self.x, self.y])
        self.body.pop()

    def turn_3(self, direction):
        if direction == 0:
            return True
        elif direction == 1:
            if self.dx < 0:
                self.dx = 0
                self.dy = cellSize
            elif self.dx > 0:
                self.dx = 0
                self.dy = -cellSize
            elif self.dy < 0:
                self.dx = -cellSize
                self.dy = 0
            elif self.dy > 0:
                self.dx = cellSize
                self.dy = 0
            return True
        elif direction == 2:
            if self.dx < 0:
                self.dx = 0
                self.dy = -cellSize
            elif self.dx > 0:
                self.dx = 0
                self.dy = cellSize
            elif self.dy < 0:
                self.dx = cellSize
                self.dy = 0
            elif self.dy > 0:
                self.dx = -cellSize
                self.dy = 0
            return True
        else:
            return False

    def turn_4(self, direction):
        if direction == 0:
            self.dx = 0
            self.dy = cellSize
        elif direction == 1:
            self.dx = 0
            self.dy = -cellSize
        elif direction == 2:
            self.dx = -cellSize
            self.dy = 0
        elif direction == 3:
            self.dx = cellSize
            self.dy = 0
        else:
            return False
        return True

    def collide(self):
        if self.body[0][0] < 0 \
                or self.body[0][0] > win_width - cellSize \
                or self.body[0][1] < 0 \
                or self.body[0][1] > win_width - cellSize \
                or self.body[0] in self.body[1:]:
            return True
        else:
            return False

    def growth(self, food):
        if self.body[0] == [food.x, food.y]:
            self.body.append(self.body[-1])
            return True
        else:
            return False

    def draw(self, win):
        for pos in range(len(self.body)):
            pygame.draw.rect(win, self.color, (self.body[pos][0], self.body[pos][1], cellSize, cellSize))

    def angle_with_food(self, food):
        food_vector = [food.x - self.body[0][0], food.y - self.body[0][1]]
        direction_vector = [self.dx, self.dy]

        dot_product = food_vector[0] * direction_vector[0] + food_vector[1] * direction_vector[1]
        food_magnitude = math.sqrt(food_vector[0] ** 2 + food_vector[1] ** 2)
        direction_magnitude = math.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2)

        if food_magnitude == 0 or direction_magnitude == 0:
            cosine_angle = 0
        else:
            cosine_angle = dot_product / (food_magnitude * direction_magnitude)

        cosine_angle = max(-1, min(cosine_angle, 1))
        angle_radians = math.acos(cosine_angle)
        angle_degrees = math.degrees(angle_radians)
        cross_product = food_vector[0] * direction_vector[1] - food_vector[1] * direction_vector[0]
        if cross_product < 0:
            angle_degrees = -angle_degrees

        return angle_degrees


class Food:
    color = (255, 0, 0)

    def __init__(self, snake):
        self.x = random.randrange(0, win_width, cellSize)
        self.y = random.randrange(0, win_height, cellSize)
        self.snake = snake

    def respawn(self):
        while True:
            self.x = random.randrange(0, win_width, cellSize)
            self.y = random.randrange(0, win_height, cellSize)
            if [self.x, self.y] not in self.snake.body:
                return True

    def draw(self, win):
        pygame.draw.ellipse(win, self.color, (self.x, self.y, cellSize, cellSize))


class Grid:
    def __init__(self):
        self.colors = [(37, 54, 69), (32, 44, 55)]
        self.color = 0

    def draw(self, win):
        for x in range(0, win_width, cellSize):
            for y in range(0, win_height, cellSize):
                if self.color > 1:
                    self.color = 0
                pygame.draw.rect(win, self.colors[self.color], (x, y, cellSize, cellSize))
                self.color += 1
            self.color -= 1


def draw_window(win, score, grid, snake, food):
    grid.draw(win)
    food.draw(win)
    snake.draw(win)
    text = stat_font.render("Score: " + str(score), 1, (255, 255, 255))
    win.blit(text, (win_width - 10 - text.get_width(), 10))
    pygame.display.update()


def get_inputs(snake, food):
    dir_l = snake.dx < 0
    dir_r = snake.dx > 0
    dir_u = snake.dy < 0
    dir_d = snake.dy > 0

    l = -cellSize
    for x in range(0, snake.body[0][0], cellSize):
        if [x, snake.body[0][1]] in snake.body:
            l = x
    r = win_width
    for x in range(win_width, snake.body[0][0], -cellSize):
        if [x, snake.body[0][1]] in snake.body:
            r = x
    u = -cellSize
    for y in range(0, snake.body[0][1], cellSize):
        if [snake.body[0][0], y] in snake.body:
            u = y
    d = win_height
    for y in range(win_height, snake.body[0][1], -cellSize):
        if [snake.body[0][0], y] in snake.body:
            d = y
    dl = (snake.body[0][0] - l) // cellSize
    dr = (r - snake.body[0][0]) // cellSize
    du = (snake.body[0][1] - u) // cellSize
    dd = (d - snake.body[0][1]) // cellSize

    inputs = np.array([
        (dir_r and dr == 1) or
        (dir_l and dl == 1) or
        (dir_u and du == 1) or
        (dir_d and dd == 1),

        (dir_u and dr == 1) or
        (dir_d and dl == 1) or
        (dir_l and du == 1) or
        (dir_r and dd == 1),

        (dir_d and dr == 1) or
        (dir_u and dl == 1) or
        (dir_r and du == 1) or
        (dir_l and dd == 1),

        # dir_l,
        # dir_r,
        # dir_u,
        # dir_d,

        snake.angle_with_food(food) / 180,
        # math.sqrt((snake.x - food.x) ** 2 + (snake.y - food.y) ** 2) / mean([win_height, win_width])
        # food.x < snake.x,  # food left
        # food.x > snake.x,  # food right
        # food.y < snake.y,  # food up
        # food.y > snake.y  # food down
    ])
    return inputs


def get_grid(snake, food, fov):
    grid = []
    for i in range(snake.y - fov * cellSize, snake.y + 1 + fov * cellSize, cellSize):
        row = []
        for j in range(snake.x - fov * cellSize, snake.x + 1 + fov * cellSize, cellSize):
            if j == food.x and i == food.y:
                row.append(2)
            else:
                found = False
                for pos in snake.body:
                    if j == pos[0] and i == pos[1]:
                        row.append(1)
                        found = True
                        break
                if not found:
                    if j < 0 or j >= win_width or i < 0 or i >= win_height:
                        row.append(1)
                    else:
                        row.append(0)

        grid.append(row)
    food_data = [food.x < snake.x,
                 food.x > snake.x,
                 food.y < snake.y,
                 food.y > snake.y]
    grid = np.array(grid).flatten()
    grid = np.append(grid, food_data)
    return grid


def main(individual: Individual, display=False):
    if display:
        win = pygame.display.set_mode((win_width, win_height))
        clock = pygame.time.Clock()
        FPS = 5
        grid = Grid()
    snake = Snake(
        win_width // 2 - cellSize * 2,
        win_height // 2 - cellSize * 2)
    food = Food(snake)
    score = 0
    max_time_alive = MAX_TIME_ALIVE
    while True:
        if display:
            clock.tick(FPS)

        inputs = get_inputs(snake, food)
        direction = individual.forward(inputs=inputs).argmax()
        snake.turn_3(direction)
        snake.move()
        if snake.collide():
            individual.bonus_fitness = -10
            return individual
        if individual.time_alive >= max_time_alive:
            individual.bonus_fitness = -10
            return individual
        if snake.growth(food):
            score += 1
            individual.score += 1
            if score < 10:
                max_time_alive += 50
            else:
                max_time_alive += 100
            food.respawn()
        # if inputs[-1] == 0:
        #     individual.bonus_fitness += 0.1
        #     max_time_alive += 1
        if display:
            draw_window(win, score, grid, snake, food)

        individual.time_alive += 1


if __name__ == '__main__':
    if not showcase:
        if os.path.exists("population.obj"):
            population = load_population("population")
            data = pd.read_csv("generation_data.csv")
            generation_number = data["Generation"].values[-1]
            print("Loaded population")
        else:
            data = pd.DataFrame(
                columns=["Generation", "Avg Fitness", "Avg score", "Max Score", "Mutations Number", "Elapsed Time"])
            population = init_population()
            generation_number = 0

        while True:
            start = time()
            generation_number += 1

            with Pool(4) as pool:
                population = pool.map(main, population)

            fitness_values = fitness_eval(population)
            avg_fitness = mean(fitness_values)
            avg_score = mean([individual.score for individual in population])

            population, mutated_number, best_individual = select_new_population(population, fitness_values)
            elapsed = time() - start
            print(
                f"Generation №{generation_number}. "
                f"Avg fitness: {avg_fitness}. "
                f"Avg score: {avg_score} "
                f"Max score: {best_individual.score} "
                f"Mutations number: {mutated_number} "
                f"Elapsed time: {elapsed}")
            data.loc[len(data.index)] = [generation_number, avg_fitness, avg_score, best_individual.score,
                                         mutated_number,
                                         elapsed]

            if generation_number % 10 == 0:
                data.to_csv("generation_data.csv", index=False)
                save_population(population, "population")
                save_individual(best_individual, "snake")
    else:
        data = pd.read_csv("generation_data.csv")
        generation_number = data["Generation"].values[-1]
        plt.plot(data["Generation"], data["Avg score"])
        plt.title("Generation average score")
        plt.xlabel("Generation")
        plt.ylabel("Generation average score")
        plt.show()
        while True:
            main(load_individual("snake.obj"), True)
