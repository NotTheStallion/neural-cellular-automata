import pygame
import sys
import copy
import random
from load_foil import *

pygame.init()


"""settings"""
matrix_size = 100
cube_size = 5

"""display part"""
display_width = (matrix_size + 2) * cube_size
display_height = (matrix_size + 2) * cube_size

display = pygame.display.set_mode((display_width, display_height))
pygame.display.set_caption("Smoke simulation")

"""fps part"""
clock = pygame.time.Clock()
FPS = 60


def draw_matrix(mat):
    display.fill((0, 0, 0))

    for y in range(len(mat)):
        for x in range(len(mat)):
            sqr = mat[y][x]
            if sqr == 1:
                pygame.draw.rect(display, (96, 96, 96), (x * cube_size, y * cube_size, cube_size, cube_size))
            elif sqr == 2:
                pygame.draw.rect(display, (255, 255, 255), (x * cube_size, y * cube_size, cube_size, cube_size))


def grow_density(mat):
    # Create a deep copy of the matrix to store the new state
    new_mat = copy.deepcopy(mat)
    
    # Iterate over each cell in the matrix, excluding the borders
    for y in range(1, len(mat)-1):
        for x in range(1, len(mat)-1):
            # Get the value of the current cell
            sqr = mat[y][x]
            
            # If the cell contains smoke (value 1)
            if sqr == 1:
                # Get the values of the neighboring cells
                u = mat[y - 1][x]
                d = mat[y + 1][x]
                r = mat[y][x + 1]
                l = mat[y][x - 1]

                # Movement: Prefer moving to the right, otherwise move randomly
                if r == 0:
                    new_mat[y][x+1] = sqr
                    new_mat[y][x] = 0
                else:
                    direction = random.choice(['up', 'down', 'left'])
                    if direction == 'left' and l == 0:
                        new_mat[y][x-1] = sqr
                        new_mat[y][x] = 0
                    elif direction == 'up' and u == 0:
                        new_mat[y-1][x] = sqr
                        new_mat[y][x] = 0
                    elif direction == 'down' and d == 0:
                        new_mat[y+1][x] = sqr
                        new_mat[y][x] = 0

                # Collision: Bounce off or merge
                if r == 1 or l == 1 or u == 1 or d == 1:
                    if random.choice([True, False]):
                        new_mat[y][x] = 0  # Merge
                    else:
                        # Bounce off
                        if direction == 'right' and r == 1:
                            new_mat[y][x-1] = sqr
                        elif direction == 'left' and l == 1:
                            new_mat[y][x+1] = sqr
                        elif direction == 'up' and u == 1:
                            new_mat[y+1][x] = sqr
                        elif direction == 'down' and d == 1:
                            new_mat[y-1][x] = sqr

    # Pressure: Move particles from high-density to low-density areas
    for y in range(1, len(mat)-1):
        for x in range(1, len(mat)-1):
            if mat[y][x] == 1:
                neighbors = [(y-1, x), (y+1, x), (y, x-1), (y, x+1)]
                low_density_neighbors = [n for n in neighbors if mat[n[0]][n[1]] == 0]
                if low_density_neighbors:
                    move_to = random.choice(low_density_neighbors)
                    new_mat[move_to[0]][move_to[1]] = 1
                    new_mat[y][x] = 0

    # Diffusion: Spread particles from high-density to low-density areas
    for y in range(1, len(mat)-1):
        for x in range(1, len(mat)-1):
            if mat[y][x] == 1:
                neighbors = [(y-1, x), (y+1, x), (y, x-1), (y, x+1)]
                for n in neighbors:
                    if mat[n[0]][n[1]] == 0:
                        new_mat[n[0]][n[1]] = 1
                        new_mat[y][x] = 0
                        break

    # Remove smoke that touches the borders
    for y in range(len(new_mat)):
        if new_mat[y][0] == 1:
            new_mat[y][0] = 0
        if new_mat[y][len(new_mat)-1] == 1:
            new_mat[y][len(new_mat)-1] = 0
    for x in range(len(new_mat[0])):
        if new_mat[0][x] == 1:
            new_mat[0][x] = 0
        if new_mat[len(new_mat)-1][x] == 1:
            new_mat[len(new_mat)-1][x] = 0

    # Return the new state of the matrix
    return new_mat


def grow(mat):
    # Create a deep copy of the matrix to store the new state
    new_mat = copy.deepcopy(mat)
    
    # Iterate over each cell in the matrix, excluding the borders
    for y in range(1, len(mat)-1):
        for x in range(1, len(mat)-1):
            # Get the value of the current cell
            sqr = mat[y][x]
            
            # If the cell contains smoke (value 1)
            if sqr == 1:
                # Get the values of the neighboring cells
                u = mat[y - 1][x]
                d = mat[y + 1][x]
                r = mat[y][x + 1]
                l = mat[y][x - 1]

                # If the cell to the right is empty, move the smoke to the right
                if r == 0:
                    new_mat[y][x+1] = sqr
                    new_mat[y][x] = 0
                else:
                    # Otherwise, randomly move the smoke up or down if those cells are empty
                    if random.randint(1, 2) == 1:
                        if new_mat[y-1][x] == 0:
                            new_mat[y-1][x] = sqr
                            new_mat[y][x] = 0
                    else:
                        if new_mat[y+1][x] == 0:
                            new_mat[y+1][x] = sqr
                            new_mat[y][x] = 0

    # Remove smoke that touches the borders
    for y in range(len(new_mat)):
        if new_mat[y][0] == 1:
            new_mat[y][0] = 0
        if new_mat[y][len(new_mat)-1] == 1:
            new_mat[y][len(new_mat)-1] = 0
    for x in range(len(new_mat[0])):
        if new_mat[0][x] == 1:
            new_mat[0][x] = 0
        if new_mat[len(new_mat)-1][x] == 1:
            new_mat[len(new_mat)-1][x] = 0

    # Return the new state of the matrix
    return new_mat


def create_matrix(path):
    matrix = [[0 for _ in range(matrix_size)] for _ in range(matrix_size)]
    
    start_idx = int(matrix_size * 0.2)
    end_idx = int(matrix_size * 0.8)
    xs, interpolated_matrix = interpolate_matrix(path, start_idx, end_idx, height=300)
    
    print(interpolated_matrix)
    
    for i in range(len(interpolated_matrix)):
        for j in range(len(interpolated_matrix[i])):
            interpolated_matrix[i][j] += matrix_size // 2
            
    print(interpolated_matrix)
    
    for i in range(matrix_size):
        for j in range(matrix_size):
            if j < start_idx or j > end_idx:
                matrix[j][i] = 0
            else:
                if i < interpolated_matrix[0][j - start_idx] and i > interpolated_matrix[1][j - start_idx]:
                    # print(f"in the {i}th line we fill from {interpolated_matrix[0][i - start_idx]} to {interpolated_matrix[1][i - start_idx]}")
                    matrix[i][j] = 2
                
    # print(matrix)
    return matrix


def game():
    global FPS
    matrix = create_matrix("data/geo05k.dat")

    # exit()

    """placing faze"""
    started = False

    color_mode = 1

    in_game = True
    while in_game:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    color_mode = 1
                if event.key == pygame.K_2:
                    color_mode = 2
                if event.key == pygame.K_ESCAPE:
                    matrix = create_matrix()
                    started = False

        """user inputs"""
        keys = pygame.key.get_pressed()
        mouse = pygame.mouse.get_pos()
        click = pygame.mouse.get_pressed(3)

        """drawing"""
        if click[0]:
            m_x, m_y = mouse[0] // cube_size, mouse[1] // cube_size
            if matrix[m_y][m_x] != 2:
                if color_mode == 1:
                    matrix[m_y][m_x] = 1
                elif color_mode == 2:
                    matrix[m_y][m_x] = 2
        """deleting"""
        if click[2]:
            m_x, m_y = mouse[0] // cube_size, mouse[1] // cube_size
            if 1 < m_x < matrix_size and 1 < m_y < matrix_size:
                matrix[m_y][m_x] = 0

        """end of start faze and mode"""
        if keys[pygame.K_SPACE]:
            started = True

        """grow matrix"""
        if started:
            matrix = grow(matrix)
            clock.tick(FPS)

        """display update"""
        draw_matrix(matrix)
        clock.tick(FPS)
        pygame.display.update()


if __name__ == '__main__':
    game()