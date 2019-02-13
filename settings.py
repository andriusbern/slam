import os, random

# Directories
MAIN_DIR       = os.path.dirname(os.path.realpath(__file__))
SCENES         = MAIN_DIR + '/scenes'
VREP_DIR       = MAIN_DIR + '/vrep'

sim_port       = random.randint(19000, 20000)
local_ip       = '127.0.0.1'


# Simulation parameters
fps = 30.               # Frames per second
simulation_steps = 5000
image_size = 800
map_size   = 40
steps_slam = 10 # Perform SLAM every n steps
