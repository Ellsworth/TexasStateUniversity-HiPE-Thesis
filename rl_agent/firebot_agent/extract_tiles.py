# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "xmltodict",
# ]
# ///

import os
import xmltodict

def extract_tiles(sdf_path):
    with open(sdf_path, 'r', encoding='utf-8') as f:
        xml_content = f.read()
    
    # Parse XML to dictionary
    data = xmltodict.parse(xml_content)
    
    # Navigate to the <include> tags in the world
    try:
        includes = data['sdf']['world']['include']
    except KeyError as e:
        print(f"Could not find expected tags: {e}")
        return []

    # Ensure it's a list (if there's only one, xmltodict returns a dict)
    if not isinstance(includes, list):
        includes = [includes]
        
    tiles = []
    
    for item in includes:
        name = item.get('name')
        if name and name.startswith('tile_'):
            pose_str = item.get('pose')
            if pose_str:
                # Pose string format: "X Y Z Roll Pitch Yaw"
                parts = pose_str.split()
                if len(parts) >= 3:
                    x, y, z = parts[0], parts[1], parts[2]
                    tiles.append({
                        'name': name,
                        'x': float(x),
                        'y': float(y),
                        'z': float(z)
                    })
                    
    return tiles

if __name__ == "__main__":
    # Script is in rl_agent/firebot_agent/, we want to reach ros2_ws/src/firebot_rl/assets/world-test.sdf
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    sdf_path = os.path.join(project_root, "ros2_ws", "src", "firebot_rl", "assets", "world-test.sdf")
    
    print(f"Reading {sdf_path}...\n")
    
    if not os.path.exists(sdf_path):
        print(f"Error: Could not find file at {sdf_path}")
        exit(1)
        
    tiles = extract_tiles(sdf_path)
    
    print(f"Extracted {len(tiles)} Tiles:")
    print("-" * 55)
    print(f"{'Tile Name':<15} | {'X':>8} | {'Y':>8} | {'Z':>8}")
    print("-" * 55)
    for tile in tiles:
        print(f"{tile['name']:<15} | {tile['x']:>8.2f} | {tile['y']:>8.2f} | {tile['z']:>8.2f}")
