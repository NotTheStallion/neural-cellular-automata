import pygame
import random

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 500, 500
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2D Airfoil Simulation")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)

# Airfoil parameters
AIRFOIL_Y = HEIGHT // 2
AIRFOIL_HEIGHT = 50
AIRFOIL_WIDTH = 150
PIXEL_SIZE = 5  # Size of each pixel

# Particle properties
PARTICLE_SIZE = PIXEL_SIZE
INITIAL_SPEED = 2

# Time parameters
FPS = 60
clock = pygame.time.Clock()

# Create airfoil as a series of pixels (simplified NACA profile)
airfoil_pixels = [
    (WIDTH // 4 + x * PIXEL_SIZE, AIRFOIL_Y + y * PIXEL_SIZE)
    for x in range(AIRFOIL_WIDTH // PIXEL_SIZE)
    for y in range(-AIRFOIL_HEIGHT // PIXEL_SIZE, AIRFOIL_HEIGHT // PIXEL_SIZE)
]

# Particle class
class Particle:
    def __init__(self, x, y, vx, vy):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy

    def update(self):
        # Move the particle
        self.x += self.vx
        self.y += self.vy

        # Reflect off screen edges
        if self.y <= 0 or self.y >= HEIGHT:
            self.vy = -self.vy

        # Detect collision with airfoil
        if (int(self.x) // PIXEL_SIZE * PIXEL_SIZE, int(self.y) // PIXEL_SIZE * PIXEL_SIZE) in airfoil_pixels:
            self.vx = -self.vx  # Reflect horizontally for simplicity

    def draw(self, surface):
        pygame.draw.rect(surface, BLUE, (int(self.x), int(self.y), PARTICLE_SIZE, PARTICLE_SIZE))


# List to store particles
particles = []

# Flag to track mouse button state
mouse_held = False

# Main loop
running = True
while running:
    screen.fill(WHITE)

    # Draw airfoil
    for pixel in airfoil_pixels:
        pygame.draw.rect(screen, RED, (pixel[0], pixel[1], PIXEL_SIZE, PIXEL_SIZE))

    # Update and draw particles
    for particle in particles:
        particle.update()
        particle.draw(screen)

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_held = True
        elif event.type == pygame.MOUSEBUTTONUP:
            mouse_held = False

    # Create particles if mouse is held down
    if mouse_held:
        mouse_x, mouse_y = pygame.mouse.get_pos()
        new_particle = Particle(
            x=mouse_x,
            y=mouse_y,
            vx=random.uniform(INITIAL_SPEED - 1, INITIAL_SPEED + 1),
            vy=random.uniform(-0.3, 0.3),
        )
        particles.append(new_particle)

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
