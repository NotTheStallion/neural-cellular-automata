
# Particle class
class Particle:
    def __init__(self, x, y, v, ):
        self.x = x
        self.y = y
        self.v = v # Velocity list along x and y axis

    def update(self):
        # Move the particle
        self.x += self.v[0]
        self.y += self.v[0]

        # Reflect off screen edges
        if self.y <= 0 or self.y >= HEIGHT:
            self.vy = -self.vy

        # Detect collision with airfoil
        if (int(self.x) // PIXEL_SIZE * PIXEL_SIZE, int(self.y) // PIXEL_SIZE * PIXEL_SIZE) in airfoil_pixels:
            self.vx = -self.vx  # Reflect horizontally for simplicity

    def draw(self, surface):
        pygame.draw.circle(surface, BLUE, (int(self.x), int(self.y)), PARTICLE_SIZE // 2)