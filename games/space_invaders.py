import pygame
import sys
import random

pygame.init()

WIDTH, HEIGHT = 800, 480
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.NOFRAME)
pygame.display.set_caption("BMO Space Invaders")

BG = (26, 59, 34)       
GREEN = (85, 255, 85)   
L_GREEN = (170, 255, 170)

fps = 60
clock = pygame.time.Clock()
font = pygame.font.SysFont("Courier New", 24, bold=True)
large_font = pygame.font.SysFont("Courier New", 48, bold=True)

class Player:
    def __init__(self):
        self.rect = pygame.Rect(WIDTH//2 - 20, HEIGHT - 50, 40, 20)
        self.speed = 5
        self.cooldown = 0

    def move(self, dx):
        self.rect.x += dx * self.speed
        self.rect.x = max(0, min(self.rect.x, WIDTH - self.rect.width))

    def draw(self, surface):
        pygame.draw.rect(surface, GREEN, self.rect)
        pygame.draw.rect(surface, GREEN, (self.rect.centerx - 5, self.rect.top - 10, 10, 10))

class Enemy:
    def __init__(self, x, y):
        self.rect = pygame.Rect(x, y, 30, 20)
        
    def draw(self, surface):
        pygame.draw.rect(surface, L_GREEN, self.rect)

class Bullet:
    def __init__(self, x, y, speed, color):
        self.rect = pygame.Rect(x - 2, y, 4, 10)
        self.speed = speed
        self.color = color

    def move(self):
        self.rect.y += self.speed
        
    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.rect)

def create_wave(level):
    enemies = []
    rows = min(5, 2 + level)
    for row in range(rows):
        for col in range(10):
            enemies.append(Enemy(100 + col * 50, 50 + row * 40))
    return enemies

player = Player()
bullets = []
enemy_bullets = []
enemies = create_wave(1)

level = 1
score = 0
enemy_dir = 1
enemy_speed = 1
enemy_move_timer = 0
state = "START"

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit(); sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit(); sys.exit()
                
            if state in ["START", "GAMEOVER"]:
                if event.key in (pygame.K_SPACE, pygame.K_RETURN):
                    player = Player()
                    bullets.clear()
                    enemy_bullets.clear()
                    level = 1
                    score = 0
                    enemies = create_wave(level)
                    state = "PLAYING"

    if state == "PLAYING":
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            player.move(-1)
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            player.move(1)
            
        if (keys[pygame.K_SPACE] or keys[pygame.K_RETURN] or keys[pygame.K_w] or keys[pygame.K_UP]) and player.cooldown == 0:
            bullets.append(Bullet(player.rect.centerx, player.rect.top, -7, GREEN))
            player.cooldown = 15
            
        if player.cooldown > 0:
            player.cooldown -= 1

        # Move bullets
        for b in bullets[:]:
            b.move()
            if b.rect.bottom < 0: bullets.remove(b)
            
        for b in enemy_bullets[:]:
            b.move()
            if b.rect.top > HEIGHT: enemy_bullets.remove(b)
            if b.rect.colliderect(player.rect):
                state = "GAMEOVER"

        # Move Enemies periodically
        enemy_move_timer += 1
        if enemy_move_timer > max(5, 30 - level * 2):
            enemy_move_timer = 0
            moved_down = False
            for e in enemies:
                e.rect.x += enemy_dir * 10
                if e.rect.right > WIDTH - 20 or e.rect.left < 20:
                    moved_down = True
                    
            if moved_down:
                enemy_dir *= -1
                for e in enemies:
                    e.rect.y += 20
                    if e.rect.bottom > player.rect.top:
                        state = "GAMEOVER"
                        
            # Alien Shooting logic
            if enemies and random.random() < 0.05 + (level * 0.01):
                shooter = random.choice(enemies)
                enemy_bullets.append(Bullet(shooter.rect.centerx, shooter.rect.bottom, 5, L_GREEN))

        # Collisions
        for b in bullets[:]:
            hit = False
            for e in enemies[:]:
                if b.rect.colliderect(e.rect):
                    enemies.remove(e)
                    if b in bullets: bullets.remove(b)
                    score += 10
                    hit = True
                    break
                    
        # Next wave
        if not enemies:
            level += 1
            enemies = create_wave(level)
            bullets.clear()
            enemy_bullets.clear()

    # Draw Logic
    screen.fill(BG)
    
    if state == "PLAYING" or state == "GAMEOVER":
        player.draw(screen)
        for e in enemies: e.draw(screen)
        for b in bullets: b.draw(screen)
        for b in enemy_bullets: b.draw(screen)
        
        score_txt = font.render(f"SCORE: {score}  WAVE: {level}", True, L_GREEN)
        screen.blit(score_txt, (10, 10))

    if state == "START":
        title = large_font.render("BMO INVADERS", True, L_GREEN)
        hint = font.render("Press SPACE or ENTER to Start!", True, GREEN)
        screen.blit(title, (WIDTH//2 - title.get_width()//2, HEIGHT//2 - 50))
        screen.blit(hint, (WIDTH//2 - hint.get_width()//2, HEIGHT//2 + 20))
        
    elif state == "GAMEOVER":
        go = large_font.render("SYSTEM FAILURE", True, L_GREEN)
        hint = font.render(f"Final Score: {score} - Press SPACE to Retry", True, GREEN)
        screen.blit(go, (WIDTH//2 - go.get_width()//2, HEIGHT//2 - 50))
        screen.blit(hint, (WIDTH//2 - hint.get_width()//2, HEIGHT//2 + 20))

    pygame.display.flip()
    clock.tick(fps)
