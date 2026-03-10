import pygame
import sys
import random

# Initialize Pygame
pygame.init()

# Setup BMO Display
WIDTH, HEIGHT = 800, 480
# Use NOFRAME to launch as a flat borderless OS window matching BMO's UI
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.NOFRAME)
pygame.display.set_caption("BMO Snake")

# BMO Retro Color Palette
BG_COLOR = (26, 59, 34)       # #1a3b22
SNAKE_COLOR = (85, 255, 85)   # #55ff55
FOOD_COLOR = (170, 255, 170)  # #aaffaa

# Game variables
CELL_SIZE = 20
grid_w = WIDTH // CELL_SIZE
grid_h = HEIGHT // CELL_SIZE

fps = 10
clock = pygame.time.Clock()

def reset_game():
    # Start snake in the middle, headed RIGHT
    snake = [(grid_w // 2, grid_h // 2), (grid_w // 2 - 1, grid_h // 2), (grid_w // 2 - 2, grid_h // 2)]
    direction = (1, 0)
    food = get_random_food(snake)
    return snake, direction, food, 0

def get_random_food(snake):
    while True:
        pos = (random.randint(0, grid_w - 1), random.randint(0, grid_h - 1))
        if pos not in snake:
            return pos

snake, direction, food, score = reset_game()

font = pygame.font.SysFont("Courier New", 24, bold=True)
large_font = pygame.font.SysFont("Courier New", 48, bold=True)

state = "PLAYING" # PLAYING or GAMEOVER

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()
            
            if state == "GAMEOVER":
                if event.key in (pygame.K_SPACE, pygame.K_RETURN):
                    snake, direction, food, score = reset_game()
                    state = "PLAYING"
            else:
                # Map BMO's natively bound arrow keys naturally to pygame
                if event.key in (pygame.K_UP, pygame.K_w) and direction[1] == 0:
                    direction = (0, -1)
                elif event.key in (pygame.K_DOWN, pygame.K_s) and direction[1] == 0:
                    direction = (0, 1)
                elif event.key in (pygame.K_LEFT, pygame.K_a) and direction[0] == 0:
                    direction = (-1, 0)
                elif event.key in (pygame.K_RIGHT, pygame.K_d) and direction[0] == 0:
                    direction = (1, 0)

    if state == "PLAYING":
        head_x, head_y = snake[0]
        new_head = (head_x + direction[0], head_y + direction[1])

        # Check wall collision 
        if new_head[0] < 0 or new_head[0] >= grid_w or new_head[1] < 0 or new_head[1] >= grid_h:
            state = "GAMEOVER"
        # Check self collision
        elif new_head in snake:
            state = "GAMEOVER"
        else:
            snake.insert(0, new_head)
            if new_head == food:
                score += 10
                fps = min(25, 10 + (score // 50)) # Speed up gradually
                food = get_random_food(snake)
            else:
                snake.pop()

    # Drawing
    screen.fill(BG_COLOR)
    
    # Draw food
    pygame.draw.rect(screen, FOOD_COLOR, (food[0]*CELL_SIZE, food[1]*CELL_SIZE, CELL_SIZE, CELL_SIZE))
    
    # Draw snake
    for i, segment in enumerate(snake):
        # Slightly darker green for the body, bright green for the head
        color = SNAKE_COLOR if i == 0 else (60, 200, 60)
        pygame.draw.rect(screen, color, (segment[0]*CELL_SIZE, segment[1]*CELL_SIZE, CELL_SIZE, CELL_SIZE))

    # Draw Score
    score_txt = font.render(f"SCORE: {score}", True, SNAKE_COLOR)
    screen.blit(score_txt, (10, 10))
    
    # Draw FPS
    fps_txt = font.render(f"P: {fps}", True, SNAKE_COLOR)
    screen.blit(fps_txt, (WIDTH - 100, 10))

    if state == "GAMEOVER":
        go_txt = large_font.render("GAME OVER", True, FOOD_COLOR)
        screen.blit(go_txt, (WIDTH//2 - go_txt.get_width()//2, HEIGHT//2 - 50))
        
        hint_txt = font.render("Press SPACE or ENTER to Replay", True, SNAKE_COLOR)
        screen.blit(hint_txt, (WIDTH//2 - hint_txt.get_width()//2, HEIGHT//2 + 20))
        
        esc_txt = font.render("Press ESCAPE to Exit", True, SNAKE_COLOR)
        screen.blit(esc_txt, (WIDTH//2 - esc_txt.get_width()//2, HEIGHT//2 + 60))

    pygame.display.flip()
    clock.tick(fps)
