#!/usr/bin/env bash
set -e
# run from workspace root (one level above src/)
# Example usage:
#   mkdir -p ~/ws/src && cd ~/ws
#   git clone https://github.com/cemkaya92/asl_rover_offboard.git src/asl_rover_offboard
#   bash src/asl_rover_offboard/bootstrap.sh
WS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$WS_ROOT"
vcs import src < src/asl_rover_offboard/deps.repos
rosdep install --from-paths src --ignore-src -r -y
