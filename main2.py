import sys
import pygame
from car import Car
import neat
from dotenv import load_dotenv
import os
import random
import torch
from neuralNetwork import NeuralNet

# Constant variables
SCREEN_HEIGHT = 1280
SCREEN_WIDTH = 720
CAR_HEIGHT = 192
CAR_WIDTH = 112
GENERATION = 0

# Definir los colores a utilizar
WHITE = pygame.Color(255, 255, 255)
BLACK = pygame.Color(0, 0, 0)
GREEN = pygame.Color(0, 255, 0)
BLUE = pygame.Color(0, 0, 255)
YELLOW = pygame.Color(255, 255, 0)
ORANGE = pygame.Color(255, 165, 0)
GREY = pygame.Color(128, 128, 128)

load_dotenv()

# Window display settings
pygame.display.set_caption('Self Driving Car!')
icon = pygame.image.load('red_car.png')
pygame.display.set_icon(icon)

reloj = pygame.time.Clock()

# game_map = pygame.image.load('track1.png')
game_map = pygame.image.load('newMap2.png')

# Init my game
pygame.init()
screen = pygame.display.set_mode((
    SCREEN_HEIGHT, SCREEN_WIDTH
))


def dibujarLinea(color=ORANGE, pointInit=(10, 10), pointEnd=(400, 400), width=10):
    pygame.draw.line(screen, color, pointInit, pointEnd, width)


def brainModel(data):
    model = NeuralNet(
        input_size=5, hidden_size1=4, hidden_size2=6, output_size=2)
    # input_tensor = torch.tensor([2.0, 3.0, 4.0, 5.0, 1.0])
    input_tensor = torch.tensor(data)
    print(data, [2.0, 3.0, 4.0, 5.0, 1.0], " esto es el dato")
    input_tensor = input_tensor.unsqueeze(0)  # convertir a dimension (1, 6)
    return model.predict(input_tensor)


def main():
    """Main method for runing the pygame window"""
    clock = pygame.time.Clock()
    # Crear una nueva superficie para dibujar la imagen desenfocada
    cars = []

    for i in range(10):
        # Init my cars
        cars.append(Car(game_map))
    # car = Car(game_map)

    contador = 0

    running = True
    while running:
        # RGB - Red, Green, Blue
        screen.fill((40, 40, 40))
        screen.blit(game_map, (0, 0))

        # ---------------------------------------------------------------
        dibujarLinea(pointInit=(SCREEN_WIDTH//2-80, SCREEN_HEIGHT),
                     pointEnd=(SCREEN_WIDTH//2-80, 520))

        for event in pygame.event.get():  # End of event loop
            if event.type == pygame.QUIT:
                running = False
        # contador +=1
        # if contador%100 == 0 : car.angle = contador
        # Input my data and get result from network
        for index, car in enumerate(cars):
            # Llamar a mi red neuronal y pasar las distancias
            # print(f' {index} {car.get_data()}')
            # i = brainModel(car.get_data())
            # print(i)
            # i = output.index(max(output)) # El maximo del vector de salida
            if 0 == 0:  # Gira derecha
                car.angle += 10
            else:      # Gira Izquierda
                car.angle -= 10

         # Update car and fitness
        remain_cars = 0
        for i, car in enumerate(cars):
            if not (car.get_collided()):
                remain_cars += 1
                car.update()
                # genomes[i][1].fitness += car.get_reward()

        # check
        if remain_cars == 0:
            break

        # Update car methods
        # car.update()

        # Drawing
        screen.blit(game_map, (0, 0))
        for car in cars:
            if not (car.get_collided()):
                car.draw(screen)

        # update display
        pygame.display.update()
        reloj.tick(60)


main()
