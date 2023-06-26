import sys
import pygame
import pygame as pg
from car import Car
import neat
from dotenv import load_dotenv
import os

# Constant variables
SCREEN_HEIGHT = 1500
SCREEN_WIDTH = 800
CAR_HEIGHT = 192
CAR_WIDTH = 112
GENERATION = 0

# Definir los colores a utilizar
WHITE = pg.Color(255, 255, 255)
BLACK = pg.Color(0, 0, 0)
GREEN = pg.Color(0, 255, 0)
BLUE = pg.Color(0, 0, 255)
YELLOW = pg.Color(255, 255, 0)
ORANGE = pg.Color(255, 165, 0)
GREY = pg.Color(128, 128, 128)
RED = pg.Color(255, 0, 0)

load_dotenv()

# Window display settings
pygame.display.set_caption('Self Driving Car!')
icon = pygame.image.load('red_car.png')
pygame.display.set_icon(icon)

# Map to be tested
env_map = os.getenv('MAP')

if env_map == '1':
    game_map = pygame.image.load('practice_track.png')
elif env_map == '2':
    game_map = pygame.image.load('track1.png')
elif env_map == '3':
    game_map = pygame.image.load('newMap001.png')
    # game_map = pygame.image.load('newMap002.png')
    # game_map = pygame.image.load('newMap003.png')
else:
    game_map = pygame.image.load('practice_track.png')


def mostrar_menu(screen):
    # Colores
    GRIS = (150, 150, 150)

    # Fuente
    fuente = pygame.font.SysFont("Arial", 30)

    # Botones
    boton1 = pygame.Rect(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 100, 200, 50)
    boton2 = pygame.Rect(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 100, 200, 50)
    boton3 = pygame.Rect(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 100, 200, 50)

    pygame.draw.rect(screen, GRIS, boton1, border_radius=10)
    pygame.draw.rect(screen, GRIS, boton2, border_radius=10)
    pygame.draw.rect(screen, GRIS, boton3, border_radius=10)

    texto1 = fuente.render("Opción 1", True, (0, 0, 0))
    texto2 = fuente.render("Opción 2", True, (0, 0, 0))
    texto3 = fuente.render("Opción 3", True, (0, 0, 0))

    screen.blit(texto1, (boton1.x + boton1.width // 2 - texto1.get_width() //
                2, boton1.y + boton1.height // 2 - texto1.get_height() // 2))
    screen.blit(texto2, (boton2.x + boton2.width // 2 - texto2.get_width() //
                2, boton2.y + boton2.height // 2 - texto2.get_height() // 2))
    screen.blit(texto3, (boton3.x + boton3.width // 2 - texto3.get_width() //
                2, boton3.y + boton3.height // 2 - texto3.get_height() // 2))

    pygame.display.flip()


def run_car(genomes, config):

    # Init NEAT
    nets = []
    cars = []

    for id, g in genomes:
        print(f'{id}, {g} ===========================================')
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0

        # Init my cars
        cars.append(Car(game_map))

    # Init my game
    pygame.init()
    screen = pygame.display.set_mode((
        SCREEN_HEIGHT, SCREEN_WIDTH
    ))

    clock = pygame.time.Clock()
    generation_font = pygame.font.SysFont("Arial", 70)
    font = pygame.font.SysFont("Arial", 30)
    # map = pygame.image.load('map.png')

    mostrar_menu(screen)  # Mostrar el menú al inicio

    # Main loop
    global GENERATION
    GENERATION += 1
    while True:
        screen.blit(game_map, (0, 0))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)
            

        # Input my data and get result from network
        for index, car in enumerate(cars):
            output = nets[index].activate(car.get_data())
            i = output.index(max(output))
            if i == 0:
                car.angle += 3  # 10
            else:
                car.angle -= 3  # 10

        # Update car and fitness
        remain_cars = 0
        for i, car in enumerate(cars):
            if not (car.get_collided()):
                remain_cars += 1
                car.update()
                genomes[i][1].fitness += car.get_reward()

        # check
        if remain_cars == 0:
            break

        # Drawing
        screen.blit(game_map, (0, 0))
        for car in cars:
            if not (car.get_collided()):
                car.draw(screen)

        text = generation_font.render(
            "Generation : " + str(GENERATION), True, GREEN)
        text_rect = text.get_rect()
        text_rect.center = (SCREEN_WIDTH + 300, 150)
        screen.blit(text, text_rect)

        text = font.render("Remain cars : " +
                           str(remain_cars), True, RED)
        text_rect = text.get_rect()
        text_rect.center = (SCREEN_WIDTH + 300, 200)
        screen.blit(text, text_rect)

        text = font.render("Number of sensors : " +
                           str(os.getenv("NUM_SENSORES")), True, GREEN)
        text_rect = text.get_rect()
        text_rect.center = (SCREEN_WIDTH + 300, 230)
        screen.blit(text, text_rect)

        pygame.display.flip()
        clock.tick(0)


if __name__ == "__main__":
    # Set configuration file
    config_path = "./config-feedforward.txt"
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    # Create core evolution algorithm class
    p = neat.Population(config)

    # Add reporter for fancy statistical result
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run NEAT
    p.run(run_car, 100)
