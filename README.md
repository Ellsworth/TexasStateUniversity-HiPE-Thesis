# TexasStateUniversity-HiPE-Thesis

RL-based FireBot navigation in Gazebo.

## Prerequisites

1.  **Podman** (or Docker) - [Installation Guide](https://podman.io/docs/installation)
2.  **NVIDIA Container Toolkit** - [Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
3.  **Just** - [Installation Guide](https://github.com/casey/just)

> [!NOTE]
> Make sure you have working NVIDIA drivers installed (`nvidia-smi`).

## Getting Started

1.  **Clone the repository**:
    ```bash
    git clone git@github.com:Ellsworth/TexasStateUniversity-HiPE-Thesis.git
    cd TexasStateUniversity-HiPE-Thesis
    ```

2.  **Launch the environment**:
    ```bash
    just
    ```
    This will set up X11 forwarding and launch the Gazebo environment.

## Common Operations

-   **`just`**: Start the Gazebo environment (default).
-   **`just train`**: Start the RL agent training.
-   **`just teleop`**: Start the teleoperation script to control the robot manually.
-   **`just shell`**: Jump into a shell within the running container.
-   **`just stop`**: Stop all running containers.
-   **`just clean`**: Remove build artifacts.
