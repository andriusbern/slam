"""
@author: Andrius Bernatavicius, 2019
"""

from __future__ import print_function
from lib.env    import VrepEnvironment
from lib.agents import Pioneer, Display
import settings, time, argparse
import matplotlib.pyplot as plt
import numpy as np

def loop(robot):
    
    """
    Agent control loop
    """
    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test', action='store_true', help='Test in a room environment')
    args = parser.parse_args()

    env = args.test
    
    # Initialize and start environment
    if env:
        environment = VrepEnvironment(settings.SCENES + '/room_d.ttt') # Open the file containing our scene (robot and its environment)
    else:
        environment = VrepEnvironment(settings.SCENES + '/room_d.ttt')  # Open the file containing our scene (robot and its environment)

    environment.start_vrep() # Start the simulator
    environment.connect()    # Connect our program to the simulator

    # Create our robot in the current environment
    robot   = Pioneer(environment)
    display = Display(robot, True)  # Display the information of the robot

    print('\nDemonstration of Simultaneous Localization and Mapping using V-REP robot simulation software. \nPress "CTRL+C" to exit.\n')

    try:
        # LOOP
        start = time.time()
        step = 0
        environment.start_simulation()
        environment.step_simulation()
        camera_handle = environment.get_handle('Vision_sensor')
        
        while step < settings.simulation_steps:

            display.update()
            environment.step_simulation()              # Advance the simulation by one step
            loop(robot)
            step += 1
            if robot.pos[0] < 330:
                end = time.time()
                print('Track completed!\nTime: {}'.format(end - start))
                environment.destroy_instances()
                break
            if robot.distance_closest()[0] < 0.25 or robot.distance_closest()[1]<.25:
                print('Hit an obstacle!\nTime: {}'.format(time.time() - start))
                environment.destroy_instances()
                break

    except KeyboardInterrupt:
        end = time.time()
        print('\n\nInterrupted! Time: {}s'.format(end-start))
        environment.destroy_instances()
    
    
