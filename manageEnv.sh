#!/usr/bin/env bash
set -e

SERVICE="gazebo"

# dl_subt → download one preset SubT model URL
if [[ "$1" == "dl_subt" ]]; then
    URL="https://fuel.gazebosim.org/1.0/OpenRobotics/collections/SubT%20Tech%20Repo"
    echo "[environmentManager] Downloading SubT preset model:"
    echo "  $URL"
    podman compose run --rm "$SERVICE" gz fuel download -u "$URL -j 16"
    exit 0
fi

# Default passthrough to gz commands
if [[ $# -eq 0 ]]; then
    echo "Usage: $0 {dl_subt|gz-subcommands...}"
    exit 1
fi

SUBCOMMAND="$1"
shift

echo "[environmentManager] Running: gz $SUBCOMMAND $*"
podman compose run --rm "$SERVICE" gz "$SUBCOMMAND" "$@"
