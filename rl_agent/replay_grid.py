import os
import sys
import time
import argparse
import numpy as np
import pygame

def main():
    recordings_dir = "recordings"
    
    parser = argparse.ArgumentParser(description="Replay obs_local_grid from RL recordings")
    parser.add_argument("-f", "--file", type=str, help="Path to the .npz file to load. Defaults to the most recent file in recordings/.")
    args = parser.parse_args()
    
    if args.file:
        npz_file = args.file
    else:
        files = sorted([f for f in os.listdir(recordings_dir) if f.endswith('.npz')])
        if not files:
            print(f"No .npz files found in {recordings_dir}/")
            sys.exit(1)
        npz_file = os.path.join(recordings_dir, files[-1]) # Default to most recent
    
    print(f"Loading {npz_file}...")
    data = np.load(npz_file)
    
    if 'obs_local_grid' not in data:
        print("Key 'obs_local_grid' not found in the file.")
        sys.exit(1)
        
    obs_local_grid = data['obs_local_grid']
    print(f"Loaded 'obs_local_grid' with shape: {obs_local_grid.shape}")
    
    # Initialize pygame
    pygame.init()
    
    num_frames = obs_local_grid.shape[0]
    # Shape is presumably (Frames, Channels, Height, Width) or (Frames, Height, Width)
    if obs_local_grid.ndim == 4:
        # e.g. (1478, 1, 65, 65)
        height = obs_local_grid.shape[2]
        width = obs_local_grid.shape[3]
    else:
        # e.g. (1478, 65, 65)
        height = obs_local_grid.shape[1]
        width = obs_local_grid.shape[2]
    
    # Scale up the display so it's not tiny (e.g. 65x65 -> 650x650)
    scale_factor = 10
    display_width = width * scale_factor
    display_height = height * scale_factor
    
    screen = pygame.display.set_mode((display_width, display_height))
    pygame.display.set_caption("obs_local_grid Replay")
    
    step_size = 0.1 # seconds per frame
    
    running = True
    frame_idx = 0
    
    while running and frame_idx < num_frames:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        # Extract the current frame
        frame = obs_local_grid[frame_idx]
        
        # Squeeze out the channel dim if it's 1
        if frame.ndim == 3 and frame.shape[0] == 1:
            frame = frame[0]
        elif frame.ndim == 3 and frame.shape[-1] == 1:
            frame = frame[:, :, 0]
            
        # Ensure the frame values are in the 0-255 range
        if frame.max() <= 1.0 and frame.max() > 0.0 and frame.dtype != np.uint8:
            display_frame = (frame * 255.0).clip(0, 255).astype(np.uint8)
        else:
            display_frame = frame.clip(0, 255).astype(np.uint8)
            
        # pygame.surfarray.make_surface expects shape (width, height, 3) for RGB
        # Our array is (height, width), so we need to transpose it to (width, height)
        display_frame = np.transpose(display_frame)
        
        # Convert grayscale to RGB format to display properly
        rgb_frame = np.stack((display_frame,) * 3, axis=-1)
        
        # Create a surface from the RGB array
        surface = pygame.surfarray.make_surface(rgb_frame)
        
        # Scale to match the display size
        scaled_surface = pygame.transform.scale(surface, (display_width, display_height))
        
        # Display the surface
        screen.blit(scaled_surface, (0, 0))
        pygame.display.flip()
        
        frame_idx += 1
        time.sleep(step_size)
        
    print("Replay finished.")
    pygame.quit()

if __name__ == "__main__":
    main()
