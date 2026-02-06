{
  description = "FireBot RL Agent Development Environment";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        
        # Define the system libraries Pygame needs to "see"
        runtimeLibs = with pkgs; [
          libGL
          xorg.libX11
          xorg.libXcursor
          xorg.libXext
          xorg.libXi
          xorg.libXinerama
          xorg.libXrandr
          xorg.libXrender
          SDL2
          SDL2_image
          SDL2_mixer
          SDL2_ttf
          stdenv.cc.cc.lib
        ];
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [ pkgs.uv pkgs.python312 ];

          shellHook = ''
            # 1. Setup the linker path so uv-installed Pygame find graphics libs
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath runtimeLibs}:$LD_LIBRARY_PATH"
            
            # 2. Prevent Python from creating .pyc files everywhere
            export PYTHONDONTWRITEBYTECODE=1
            
            # 3. Create the venv if it doesn't exist
            if [ ! -d ".venv" ]; then
              uv venv
            fi
            
            echo "🔥 FireBot Environment Active"
            echo "Run 'uv run teleop_gym.py' to start."
          '';
        };
      });
}
