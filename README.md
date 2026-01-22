# TexasStateUniversity-HiPE-Thesis

## Installing on NVIDIA

1. Make sure you have working NVIDIA drivers ```nvidia-smi```
2. Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html)
3. Install [Podman](https://podman.io/) or [Docker](docker.com). If using Docker, a [rootless install is recommended](https://docs.docker.com/engine/security/rootless/).
4. Make sure you have ```podman-compose``` or ```docker-compose``` installed.
5. Clone this repository ```git clone git@github.com:Ellsworth/TexasStateUniversity-HiPE-Thesis.git```
6. ```cd TexasStateUniversity-HiPE-Thesis```
7. Launch it! ```podman compose up```

## Installing on AMD
1. Install [Podman](https://podman.io/) or [Docker](docker.com). If using Docker, a [rootless install is recommended](https://docs.docker.com/engine/security/rootless/).
2. Make sure you have ```podman-compose``` or ```docker-compose``` installed.
3. Clone this repository ```git clone git@github.com:Ellsworth/TexasStateUniversity-HiPE-Thesis.git```
4. ```cd TexasStateUniversity-HiPE-Thesis```
5. Launch it! ```podman compose up```