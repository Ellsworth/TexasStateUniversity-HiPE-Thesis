import gymnasium as gym
import pygame
import numpy as np
import argparse
import sys
import time
from firebot_agent.gym_env import FireBotEnv

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 900
GRID_SIZE = 65
SCALE_FACTOR = 10  # Scale up grid for visibility (65 -> 650 pixels)
OFFSET_X = (WINDOW_WIDTH - (GRID_SIZE * SCALE_FACTOR)) // 2
OFFSET_Y = 50

# Colors
COLOR_BG = (30, 30, 30)
COLOR_TEXT = (255, 255, 255)
COLOR_GRID_UNKNOWN = (127, 127, 127)
COLOR_GRID_FREE = (255, 255, 255)
COLOR_GRID_OCCUPIED = (0, 0, 0)

class FireBotTeleop:
    def __init__(self, continuous=True, mock=False):
        self.continuous = continuous
        self.mock = mock
        
        # Initialize Environment
        self.env = FireBotEnv(discrete_actions=not continuous, mock=mock, agent_name="teleop")
        self.observation, _ = self.env.reset()
        
        # Pygame Setup
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("FireBot Teleop")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 24)
        
        # State
        self.running = True
        self.action = np.zeros(2, dtype=np.float32) if continuous else 0
        self.total_reward = 0.0
        self.step_count = 0
        
        print("\nControls:")
        print("  Arrow Keys / WASD: Move")
        print("  Space: Stop")
        print("  ESC / Q: Quit")
        print("-" * 30)

    def get_action_from_keys(self):
        keys = pygame.key.get_pressed()
        
        linear_x = 0.0
        angular_z = 0.0
        
        # Linear Movement
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            linear_x += 1.0
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            linear_x -= 1.0
            
        # Angular Movement
        if keys[pygame.K_LEFT] or keys[pygame.K_d]:
            angular_z -= 1.0
        if keys[pygame.K_RIGHT] or keys[pygame.K_a]:
            angular_z += 1.0
            
        if self.continuous:
            return np.array([linear_x, angular_z], dtype=np.float32)
        else:
            # Simple discrete mapping (logic matches env action map somewhat)
            if linear_x > 0: return 1 # Forward
            if linear_x < 0: return 2 # Backward
            if angular_z > 0: return 3 # Left
            if angular_z < 0: return 4 # Right
            return 0 # Stop

    def render_grid(self, local_grid):
        # local_grid shape is (1, 65, 65)
        # Squeeze to (65, 65)
        grid = np.squeeze(local_grid)
        
        # Create surface for the grid
        surf = pygame.Surface((GRID_SIZE, GRID_SIZE))
        
        # Create RGB array
        # This is a bit slow doing it manually per pixel in python loop, 
        # but for 65x65 it's instant.
        # Faster way: use pygame.surfarray
        
        rgb_array = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.uint8)
        
        # Map values to colors
        # 0 -> Free (White)
        # 127 -> Unknown (Grey)
        # 255 -> Occupied (Black) -- wait, in env 255 was occupied? 
        # Let's check env.py: 
        #   processed_grid[mask_free] = 0
        #   processed_grid[mask_occupied] = val * 2.55 (so 255 is strong occupied)
        # Let's invert for display: 255 (occupied) -> Black, 0 (free) -> White
        
        # Actually simplest to just display grayscale
        # If 0 is free (white in occupancy grid logic usually), we want 255 for display
        # If 255 is occupied (black), we want 0 for display
        
        display_grid = 255 - grid
        
        # Duplicate to 3 channels by stacking
        rgb_array = np.dstack((display_grid, display_grid, display_grid))
        
        # Transpose for Pygame (width, height) vs numpy (row, col)
        # Usually numpy image is (y, x), pygame surface is (x, y)
        rgb_array = np.transpose(rgb_array, (1, 0, 2))
        rgb_array = np.flipud(rgb_array)
        
        # Blit array to surface
        pygame.surfarray.blit_array(surf, rgb_array)
        
        # Scale up
        scaled_surf = pygame.transform.scale(surf, (GRID_SIZE * SCALE_FACTOR, GRID_SIZE * SCALE_FACTOR))
        
        # Draw to screen
        self.screen.blit(scaled_surf, (OFFSET_X, OFFSET_Y))
        
        # Draw border
        pygame.draw.rect(self.screen, (255, 0, 0), (OFFSET_X, OFFSET_Y, GRID_SIZE * SCALE_FACTOR, GRID_SIZE * SCALE_FACTOR), 2)
        
        # Draw Robot Arrow (Center of Grid, pointing UP)
        center_x = OFFSET_X + (GRID_SIZE * SCALE_FACTOR) // 2
        center_y = OFFSET_Y + (GRID_SIZE * SCALE_FACTOR) // 2
        arrow_size = 20
        
        # Points for upward pointing arrow
        p1 = (center_x, center_y - arrow_size)         # Top
        p2 = (center_x - arrow_size // 2, center_y + arrow_size // 2) # Bottom Left
        p3 = (center_x, center_y + arrow_size // 4)    # Bottom Center (Indent)
        p4 = (center_x + arrow_size // 2, center_y + arrow_size // 2) # Bottom Right
        
        pygame.draw.polygon(self.screen, (0, 100, 255), [p1, p2, p3, p4])
    def draw_text(self, text, x, y, color=COLOR_TEXT):
        text_surface = self.font.render(text, True, color)
        self.screen.blit(text_surface, (x, y))

    def run(self):
        while self.running:
            # 1. Event Handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                        self.running = False
                    if event.key == pygame.K_r:
                        print("Resetting environment...")
                        self.observation, _ = self.env.reset()
                        self.total_reward = 0.0
                        self.step_count = 0
            
            # 2. Get User Action
            self.action = self.get_action_from_keys()
            
            # 3. Step Environment
            obs, reward, terminated, truncated, info = self.env.step(self.action)
            
            self.total_reward += reward
            self.step_count += 1
            
            # 4. Rendering
            self.screen.fill(COLOR_BG)
            
            # Render Grid
            if "local_grid" in obs:
                self.render_grid(obs["local_grid"])
            
            # Render Info Text
            wall_dist = obs["wall_distance"][0]
            wall_ang = obs["wall_angle"][0]
            
            text_x = 50
            text_y_start = OFFSET_Y + (GRID_SIZE * SCALE_FACTOR) + 20
            
            self.draw_text(f"Step: {self.step_count}", text_x, text_y_start)
            self.draw_text(f"Action: {self.action}", text_x, text_y_start + 30)
            self.draw_text(f"Wall Dist: {wall_dist:.2f} m", text_x, text_y_start + 60)
            self.draw_text(f"Wall Angle: {wall_ang:.2f} rad", text_x, text_y_start + 90)
            self.draw_text(f"Reward: {reward:.4f} (Total: {self.total_reward:.2f})", text_x, text_y_start + 120)
            
            pygame.display.flip()
            
            # Clock tick
            self.clock.tick(20) # 20 Hz teleop
            
        self.env.close()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Teleop for FireBot Gym Env")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode (no ZMQ)")
    parser.add_argument("--discrete", action="store_true", help="Use discrete action space")
    
    args = parser.parse_args()
    
    teleop = FireBotTeleop(continuous=not args.discrete, mock=args.mock)
    teleop.run()
