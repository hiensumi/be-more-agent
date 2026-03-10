import pygame
import sys
import random
import os

pygame.init()

# Setup BMO Display
WIDTH, HEIGHT = 800, 480
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.NOFRAME)
pygame.display.set_caption("Flappy BMO")

# Retro Color Palette
BG_COLOR = (26, 59, 34)       # #1a3b22
PIPE_COLOR = (85, 255, 85)    # #55ff55
TEXT_COLOR = (170, 255, 170)  # #aaffaa

fps = 60
clock = pygame.time.Clock()
font = pygame.font.SysFont("Courier New", 24, bold=True)
large_font = pygame.font.SysFont("Courier New", 48, bold=True)

# Load BMO Face 
bmo_face_path = os.path.join(os.path.dirname(__file__), "..", "faces", "idle", "idle-0.png")
if os.path.exists(bmo_face_path):
    # Scale down BMO's full 800x480 face so it acts as the flappy bird!
    face_img = pygame.image.load(bmo_face_path).convert_alpha()
    bmo_rect = face_img.get_rect()
    scale = 60 / bmo_rect.width 
    bmo_img = pygame.transform.scale(face_img, (int(bmo_rect.width * scale), int(bmo_rect.height * scale)))
else:
    # Fallback to a plain green square if the image is missing
    bmo_img = pygame.Surface((40, 30))
    bmo_img.fill(TEXT_COLOR)

# Game Variables
gravity = 0.5
flap_strength = -8
pipe_width = 70
pipe_gap = 150
pipe_velocity = -4

def reset_game():
    bmo_x = int(WIDTH * 0.2)
    bmo_y = int(HEIGHT * 0.5)
    velocity = 0
    pipes = []
    score = 0
    return bmo_x, bmo_y, velocity, pipes, score

def spawn_pipe(pipes):
    min_height = 50
    max_height = HEIGHT - pipe_gap - min_height
    top_height = random.randint(min_height, max_height)
    
    # Top pipe, Bottom pipe, Passed boolean
    pipes.append([
        pygame.Rect(WIDTH, 0, pipe_width, top_height),
        pygame.Rect(WIDTH, top_height + pipe_gap, pipe_width, HEIGHT - top_height - pipe_gap),
        False
    ])

bmo_x, bmo_y, velocity, pipes, score = reset_game()
spawn_pipe(pipes)

state = "START" 

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()
            
            # Map BMO's natively bound jump keys (Space, Enter, Up, W)
            if event.key in (pygame.K_SPACE, pygame.K_RETURN, pygame.K_UP, pygame.K_w):
                if state == "START":
                    state = "PLAYING"
                    velocity = flap_strength
                elif state == "PLAYING":
                    velocity = flap_strength
                elif state == "GAMEOVER":
                    bmo_x, bmo_y, velocity, pipes, score = reset_game()
                    spawn_pipe(pipes)
                    state = "START"

    if state == "PLAYING":
        # Bird physics
        velocity += gravity
        bmo_y += velocity

        # Pipe movement and spawning
        for pipe in pipes:
            pipe[0].x += pipe_velocity
            pipe[1].x += pipe_velocity
            
            # Check if bird passed the pipe
            if pipe[0].right < bmo_x and not pipe[2]:
                score += 1
                pipe[2] = True
        
        # Remove off-screen pipes
        if pipes and pipes[0][0].right < 0:
            pipes.pop(0)
            
        # Spawn new pipes continuously
        if pipes and pipes[-1][0].x < WIDTH - 300:
            spawn_pipe(pipes)

        # Collision detection (Ceiling, Floor, Pipes)
        bmo_rect = bmo_img.get_rect(center=(bmo_x, int(bmo_y)))
        
        if bmo_y > HEIGHT or bmo_y < 0:
            state = "GAMEOVER"
            
        for pipe in pipes:
            if bmo_rect.colliderect(pipe[0]) or bmo_rect.colliderect(pipe[1]):
                state = "GAMEOVER"

    # Drawing
    screen.fill(BG_COLOR)
    
    # Draw pipes
    for pipe in pipes:
        pygame.draw.rect(screen, PIPE_COLOR, pipe[0])
        pygame.draw.rect(screen, PIPE_COLOR, pipe[1])
        
    # Draw BMO
    bmo_rect = bmo_img.get_rect(center=(bmo_x, int(bmo_y)))
    # Rotate BMO slightly based on velocity to make it look like he's flying
    rotated_bmo = pygame.transform.rotate(bmo_img, -velocity * 3)
    rotated_rect = rotated_bmo.get_rect(center=bmo_rect.center)
    screen.blit(rotated_bmo, rotated_rect.topleft)

    # Draw Text
    score_txt = font.render(f"SCORE: {score}", True, TEXT_COLOR)
    screen.blit(score_txt, (10, 10))

    if state == "START":
        start_txt = large_font.render("FLAPPY BMO", True, TEXT_COLOR)
        hint_txt = font.render("Press SPACE or UP to Flap!", True, PIPE_COLOR)
        hint2_txt = font.render("(Press ESCAPE to Exit)", True, PIPE_COLOR)
        screen.blit(start_txt, (WIDTH//2 - start_txt.get_width()//2, HEIGHT//2 - 60))
        screen.blit(hint_txt, (WIDTH//2 - hint_txt.get_width()//2, HEIGHT//2 + 10))
        screen.blit(hint2_txt, (WIDTH//2 - hint2_txt.get_width()//2, HEIGHT//2 + 40))
        
    elif state == "GAMEOVER":
        go_txt = large_font.render("GAME OVER", True, TEXT_COLOR)
        hint_txt = font.render("Press SPACE or UP to Replay!", True, PIPE_COLOR)
        hint2_txt = font.render("(Press ESCAPE to Exit)", True, PIPE_COLOR)
        screen.blit(go_txt, (WIDTH//2 - go_txt.get_width()//2, HEIGHT//2 - 60))
        screen.blit(hint_txt, (WIDTH//2 - hint_txt.get_width()//2, HEIGHT//2 + 10))
        screen.blit(hint2_txt, (WIDTH//2 - hint2_txt.get_width()//2, HEIGHT//2 + 40))

    pygame.display.flip()
    clock.tick(fps)
