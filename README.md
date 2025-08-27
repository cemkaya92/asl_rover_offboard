## Build (source)
```bash
mkdir -p ~/ws/src && cd ~/ws
git clone https://github.com/cemkaya92/asl_rover_offboard.git
vcs import src < src/asl_rover_offboard/deps.repos
rosdep install --from-paths src --ignore-src -r -y
colcon build
source install/setup.bash
