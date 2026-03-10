import pygame
import sys
import random

pygame.init()

# Setup BMO Display
WIDTH, HEIGHT = 800, 480
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.NOFRAME)
pygame.display.set_caption("BMO Pong")

# Retro Color Palette
BG_COLOR = (26, 59, 34)       # #1a3b22
ELEMENT_COLOR = (85, 255, 85) # #55ff55
TEXT_COLOR = (170, 255, 170)  # #aaffaa

fps = 60
clock = pygame.time.Clock()
font = pygame.font.SysFont("Courier New", 48, bold=True)
small_font = pygame.font.SysFont("Courier New", 24, bold=True)

# Game entities
paddle_w, paddle_h = 20, 100
ball_size = 20

def reset_ball():
    x = WIDTH // 2 - ball_size // 2
    y = HEIGHT // 2 - ball_size // 2
    speed_x = random.choice([-5, 5])
    speed_y = random.choice([-5, 5])
    return pygame.Rect(x, y, ball_size, ball_size), speed_x, speed_y

player_rect = pygame.Rect(30, HEIGHT // 2 - paddle_h // 2, paddle_w, paddle_h)
ai_rect = pygame.Rect(WIDTH - 30 - paddle_w, HEIGHT // 2 - paddle_h // 2, paddle_w, paddle_h)
ball_rect, ball_speed_x, ball_speed_y = reset_ball()

player_score = 0
ai_score = 0
paddle_speed = 7

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
            
            if state == "START" or state == "GAMEOVER":
                if event.key in (pygame.K_SPACE, pygame.K_RETURN):
                    if state == "GAMEOVER":
                        player_score = 0
                        ai_score = 0
                    state = "PLAYING"
                    ball_rect, ball_speed_x, ball_speed_y = reset_ball()

    if state == "PLAYING":
        keys = pygame.key.get_pressed()
        
        # Player 1 controls (Up/W and Down/S)
        if (keys[pygame.K_UP] or keys[pygame.K_w]) and player_rect.top > 0:
            player_rect.y -= paddle_speed
        if (keys[pygame.K_DOWN] or keys[pygame.K_s]) and player_rect.bottom < HEIGHT:
            player_rect.y += paddle_speed
            
        # Basic AI
        if ai_rect.centery < ball_rect.centery and ai_rect.bottom < HEIGHT:
            ai_rect.y += paddle_speed - 2
        elif ai_rect.centery > ball_rect.centery and ai_rect.top > 0:
            ai_rect.y -= paddle_speed - 2

        # Ball movement
        ball_rect.x += ball_speed_x
        ball_rect.y += ball_speed_y

        # Ball collision (Top/Bottom walls)
        if ball_rect.top <= 0 or ball_rect.bottom >= HEIGHT:
            ball_speed_y *= -1

        # Ball collision (Paddles)
        if ball_rect.colliderect(player_rect):
            ball_speed_x = abs(ball_speed_x) + 0.5  # Speed up slightly
            ball_rect.left = player_rect.right
        elif ball_rect.colliderect(ai_rect):
            ball_speed_x = -abs(ball_speed_x) - 0.5
            ball_rect.right = ai_rect.left

        # Scoring
        if ball_rect.left <= 0:
            ai_score += 1
            ball_rect, ball_speed_x, ball_speed_y = reset_ball()
        elif ball_rect.right >= WIDTH:
            player_score += 1
            ball_rect, ball_speed_x, ball_speed_y = reset_ball()
            
        if player_score >= 10 or ai_score >= 10:
            state = "GAMEOVER"

    # Drawing
    screen.fill(BG_COLOR)
    
    # Center dashed line
    for y in range(0, HEIGHT, 40):
        pygame.draw.rect(screen, ELEMENT_COLOR, (WIDTH//2 - 2, y, 4, 20))

    if state == "PLAYING" or state == "GAMEOVER":
        pygame.draw.rect(screen, ELEMENT_COLOR, player_rect)
        pygame.draw.rect(screen, ELEMENT_COLOR, ai_rect)
        pygame.draw.rect(screen, ELEMENT_COLOR, ball_rect)
        
        score_text = font.render(f"{player_score}   {ai_score}", True, TEXT_COLOR)
        screen.blit(score_text, (WIDTH//2 - score_text.get_width()//2, 20))

    if state == "START":
        title = font.render("BMO PONG", True, TEXT_COLOR)
        hint = small_font.render("Press SPACE or ENTER to Serve!", True, ELEMENT_COLOR)
        screen.blit(title, (WIDTH//2 - title.get_width()//2, HEIGHT//2 - 50))
        screen.blit(hint, (WIDTH//2 - hint.get_width()//2, HEIGHT//2 + 20))
        
    elif state == "GAMEOVER":
        msg = "YOU WIN!" if player_score > ai_score else "AI WINS!"
        go_txt = font.render(msg, True, TEXT_COLOR)
        hint_txt = small_font.render("Press SPACE or ENTER to Replay", True, ELEMENT_COLOR)
        screen.blit(go_txt, (WIDTH//2 - go_txt.get_width()//2, HEIGHT//2 - 50))
        screen.blit(hint_txt, (WIDTH//2 - hint_txt.get_width()//2, HEIGHT//2 + 20))

    pygame.display.flip()
    clock.tick(fps)
