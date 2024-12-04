import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Simulation parameters
GRID_SIZE = 500  # Size of the simulation grid
TIME_STEP = 1  # Time step for simulation
ITERATIONS = 20  # Gauss-Seidel iterations
GRAVITY = np.array([0, 9.81])  # Gravity vector
RELAXATION = 1.9  # Over-relaxation factor

# Initialize the velocity field (u, v)
velocity = np.zeros((GRID_SIZE, GRID_SIZE, 2))  # [u, v] for each cell
density = np.zeros((GRID_SIZE, GRID_SIZE))  # Smoke density

# Define obstacles
obstacles = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
center = (250, 250)
radius = 50
for i in range(GRID_SIZE):
    for j in range(GRID_SIZE):
        if (i - center[0])**2 + (j - center[1])**2 <= radius**2:
            obstacles[i, j] = True

# Helper functions
def apply_gravity(velocity, timestep):
    """Apply gravity to the velocity field."""
    velocity[:, :, 1] += GRAVITY[1] * timestep
    velocity[obstacles] = 0  # Zero velocity in obstacle cells

def divergence(velocity):
    """Compute the divergence of the velocity field."""
    div = np.zeros((GRID_SIZE, GRID_SIZE))
    div[:-1, :] += velocity[1:, :, 0] - velocity[:-1, :, 0]  # ∂u/∂x
    div[:, :-1] += velocity[:, 1:, 1] - velocity[:, :-1, 1]  # ∂v/∂y
    div[obstacles] = 0  # Zero divergence in obstacle cells
    return div

def project(velocity):
    """Project the velocity field to make it incompressible."""
    pressure = np.zeros((GRID_SIZE, GRID_SIZE))
    div = divergence(velocity)
    for _ in range(ITERATIONS):
        pressure[1:-1, 1:-1] = (
            (pressure[:-2, 1:-1] + pressure[2:, 1:-1] +
             pressure[1:-1, :-2] + pressure[1:-1, 2:] - div[1:-1, 1:-1])
            / 4
        )
        # Boundary conditions
        pressure[0, :] = pressure[-1, :] = pressure[:, 0] = pressure[:, -1] = 0
        pressure[obstacles] = 0  # Zero pressure in obstacle cells

    # Subtract pressure gradient from velocity
    velocity[1:, :, 0] -= np.diff(pressure, axis=0)
    velocity[:, 1:, 1] -= np.diff(pressure, axis=1)
    velocity[obstacles] = 0  # Zero velocity in obstacle cells

def semi_lagrangian(velocity, density, timestep):
    """Perform semi-Lagrangian advection."""
    new_density = np.zeros_like(density)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if obstacles[i, j]:
                continue  # Skip obstacle cells

            # Trace backward
            x = i - timestep * velocity[i, j, 0]
            y = j - timestep * velocity[i, j, 1]

            # Clamp to grid bounds
            x = max(0, min(GRID_SIZE - 1, x))
            y = max(0, min(GRID_SIZE - 1, y))

            # Bilinear interpolation
            x0, y0 = int(x), int(y)
            x1, y1 = min(x0 + 1, GRID_SIZE - 1), min(y0 + 1, GRID_SIZE - 1)
            dx, dy = x - x0, y - y0

            new_density[i, j] = (
                (1 - dx) * (1 - dy) * density[x0, y0] +
                dx * (1 - dy) * density[x1, y0] +
                (1 - dx) * dy * density[x0, y1] +
                dx * dy * density[x1, y1]
            )
    return new_density

# Simulation loop
def update(frame):
    global velocity, density
    apply_gravity(velocity, TIME_STEP)
    project(velocity)
    density = semi_lagrangian(velocity, density, TIME_STEP)
    density[200:300, 0] += 10  # Add smoke source
    im.set_array(density)

# Visualization setup
fig, ax = plt.subplots()
im = ax.imshow(density, cmap="hot", origin="lower", vmin=0, vmax=100)
# Plot obstacles
for i in range(GRID_SIZE):
    for j in range(GRID_SIZE):
        if obstacles[i, j]:
            ax.add_patch(plt.Circle((j, i), 0.5, color='black'))
ani = FuncAnimation(fig, update, frames=100, interval=10)
plt.show()
