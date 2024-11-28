import pygame
import sys
import copy
import random

pygame.init()


"""settings"""
matrix_size = 50
cube_size = 10

"""display part"""
display_width = (matrix_size + 2) * cube_size
display_height = (matrix_size + 2) * cube_size

display = pygame.display.set_mode((display_width, display_height))
pygame.display.set_caption("Water And Lava")

"""fps part"""
clock = pygame.time.Clock()
FPS = 120


def draw_matrix(mat):
    display.fill((0, 0, 0))
    print(mat)
    for y in range(len(mat)):
        for x in range(len(mat)):
            sqr = mat[y][x]
            if sqr == 1:
                pygame.draw.rect(display, (11, 87, 217), (x * cube_size, y * cube_size, cube_size, cube_size))
            elif sqr == 2:
                pygame.draw.rect(display, (240, 130, 12), (x * cube_size, y * cube_size, cube_size, cube_size))
            elif sqr == 3:
                pygame.draw.rect(display, (114, 96, 89), (x * cube_size, y * cube_size, cube_size, cube_size))


def grow(mat):
    new_mat = copy.deepcopy(mat)
    for y in range(1, len(mat)):
        for x in range(1, len(mat)):
            sqr = mat[y][x]
            if sqr == 1 or sqr == 2:

                u = mat[y - 1][x]
                d = mat[y + 1][x]
                r = mat[y][x + 1]
                l = mat[y][x - 1]

                if sqr == 1:
                    if d == 2:
                        new_mat[y+1][x] = 3
                        new_mat[y][x] = 3
                    if l == 2:
                        new_mat[y][x-1] = 3
                        new_mat[y][x] = 3
                    if r == 2:
                        new_mat[y][x+1] = 3
                        new_mat[y][x] = 3
                    if u == 2:
                        new_mat[y-1][x] = 3
                        new_mat[y][x] = 3

                if d == 0:
                    new_mat[y+1][x] = sqr
                    new_mat[y][x] = 0
                else:
                    if random.randint(1, 2) == 1:
                        if new_mat[y][x-1] == 0:
                            new_mat[y][x-1] = sqr
                            new_mat[y][x] = 0
                    else:
                        if new_mat[y][x+1] == 0:
                            new_mat[y][x+1] = sqr
                            new_mat[y][x] = 0

    return new_mat


def create_matrix():
    matrix = [[3] * (matrix_size + 2)]
    matrix += [[3] + ([0] * matrix_size) + [3] for _ in range(matrix_size)]
    matrix += [[3] * (matrix_size + 2)]
    return matrix


def game():
    global FPS
    matrix = create_matrix()

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
                if event.key == pygame.K_3:
                    color_mode = 3
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
            if matrix[m_y][m_x] != 3:
                if color_mode == 1:
                    matrix[m_y][m_x] = 1
                elif color_mode == 2:
                    matrix[m_y][m_x] = 2
                elif color_mode == 3:
                    matrix[m_y][m_x] = 3
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