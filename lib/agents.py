"""
@author: Andrius Bernatavicius, 2019

"""
from lib.algorithms import RMHC_SLAM
from lib.sensors import Laser
import numpy as np
import settings
from collections import deque
import cv2


class Pioneer(object):
    def __init__(self, env):
        
        self.env = env
        
        # Lidar
        self.lidar_names   = ['SICK_TiM310_sensor1', 'SICK_TiM310_sensor2']
        for _ in range(10):
            self.lidar_handles = [self.env.get_handle(x) for x in self.lidar_names]
        self.lidar_data    = []
        self.lidar_angle   = 270.
        self.lidar_points  = 134
        self.lidar         = Laser(self.lidar_points, 2, self.lidar_angle, 10, 2, 50)
        self.scale_factor  = 1000
        # SLAM
        self.slam_engine = RMHC_SLAM(self.lidar, settings.image_size, settings.map_size)

        # Motors
        self.motor_names   = ['Pioneer_p3dx_leftMotor', 'Pioneer_p3dx_rightMotor']
        self.motor_handles = [self.env.get_handle(x) for x in self.motor_names]

        # Positions and angles
        self.angular_velocity = np.zeros(2)
        self.angles = np.zeros(2)
        self.theta  = 0
        self.pos    = [0, 0]
        self.position_history = deque(maxlen=100)

        self.wall_l = False
        self.wall_r = False

        self.change_velocity([0, 0])


    def find_closest(self):
        """
        Returns [[angle_l, angle_r], [distance_l, distance_r]] of the closest objects in the left and right visual fields
        """
        sensor_scan_angle = self.lidar_angle / self.lidar_points
        left  = self.lidar_data[0:68]
        right = self.lidar_data[68:]
        min_left  = min(left)
        min_right = min(right)
        ind_left  = [i for i, x in enumerate(left) if x == min_left][0]
        ind_right = [i for i, x in enumerate(right) if x == min_right][0]
        angle_l = -self.lidar_angle / 2 + ind_left * sensor_scan_angle
        angle_r = 0 + ind_right * sensor_scan_angle
        
        return [[angle_l, angle_r], [min_left, min_right]]

    def angle_closest(self):
        return self.find_closest()[0]

    def distance_closest(self):
        return self.find_closest()[1]

    def wall_left(self, threshold):
        
        if sum(self.lidar_data[47:67]) < threshold*60:
            self.wall_l = True
        else:
            self.wall_l = False
        return self.wall_l

    def wall_right(self, threshold):
        if sum(self.lidar_data[67:87]) < threshold*60:
            self.wall_r = True
        else:
            self.wall_r = False
        return self.wall_r


    def read_lidars(self, mode='blocking'):
        """
        Read the vision sensor in VREP
        """
        self.lidar_data = []
        lidar_data = [self.env.read_lidar(x, mode)[2][1] for x in self.lidar_handles]
        self.lidar_data = lidar_data[0][1::2][0::4] + lidar_data[1][1::2][0::4]
        del self.lidar_data[68]
        del self.lidar_data[0]
        return self.lidar_data

    def slam(self, bytearray):
        """
        Get input from sensors and perform SLAM
        """

        # Mapping
        scan = self.read_lidars()
        scan = list(np.array(scan) * self.scale_factor)
        self.slam_engine.update(scan)
        self.slam_engine.getmap(bytearray) # Draw current map on the bytearray

        # Localization
        x, y, theta = self.slam_engine.getpos()
        self.pos[0] = int(x / float(settings.map_size * 1000) * settings.image_size)
        self.pos[1] = int(y / float(settings.map_size * 1000) * settings.image_size)
        self.position_history.append(tuple(self.pos)) # Append the position deque with a tuple of (x,y)
        self.theta = theta


    def current_speed(self):
        prev_angles = np.copy(self.angles)
        self.angles = np.array([self.env.get_joint_angle(x) for x in self.motor_handles])
        angular_velocity = self.angles - prev_angles
        for i, v in enumerate(angular_velocity):
            # In case radians reset to 0
            if v < -np.pi:
                angular_velocity[i] =  np.pi*2 + angular_velocity[i]
            if v > np.pi:
                angular_velocity[i] = -np.pi*2 + angular_velocity[i]
        self.angular_velocity = angular_velocity
        return self.angular_velocity

    def change_velocity(self, velocities, target=None):
        # Change the current velocity of the robot's wheels
        if target == 'left':
            self.env.set_target_velocity(self.motor_handles[0], velocities)
        if target == 'right':
            self.env.set_target_velocity(self.motor_handles[1], velocities)
        else:
            for i in range(2):
                self.env.set_target_velocity(self.motor_handles[i], velocities[i])
        

class Display(object):
    def __init__(self, agent, wall):
        self.bytearray = bytearray(settings.image_size*settings.image_size)
        self.agent = agent
        self.im = None
        self.colormap = cv2.COLORMAP_OCEAN
        self.visited = np.ones([settings.image_size, settings.image_size, 3])

        self.wall_disp = wall
        # Agent parameters
        self.agent_radius    = 10
        self.agent_color     = (100, 100, 100)

        # Speed display
        self.speed_location  = [(60,700), (90,700)]
        self.step = 0
        # Create cv2 window
        cv2.namedWindow('Simultaneous Localization and Mapping (SLAM)', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Simultaneous Localization and Mapping (SLAM)', 850, 850)


    def update(self):
        """
        Updates the current display based on current information of the agent
        """

        if self.step % settings.steps_slam == 0:
            self.agent.slam(self.bytearray)

        self.im = self.to_image()
        self.draw_agent(self.im)      
        
        # self.im = cv2.cvtColor(cv2.Canny(self.im, 100,150), cv2.COLOR_GRAY2RGB)
        self.draw_closest(self.im)
        self.draw_trajectory(self.im)
        self.im = cv2.flip(self.im, 0) 

        self.draw_elements(self.im)        
        self.draw_speed(self.im)
        # im = cv2.filter2D(im,-1,np.ones((5,5),np.float32)/25)
        self.im = cv2.blur(self.im,(3,3))
        cv2.imshow('Simultaneous Localization and Mapping (SLAM)', self.im)
        key = cv2.waitKey(1) & 0xFF
        self.step += 1


    def draw_closest(self, image):
        """
        Draws lines to the closest objects right and left
        """
        angles, distances = self.agent.find_closest()
        scale = 30
        x = self.agent.pos[0] + int(np.cos((self.agent.theta+angles[0])*np.pi/180) * distances[0]*scale) 
        y = self.agent.pos[1] + int(np.sin((self.agent.theta+angles[0])*np.pi/180) * distances[0]*scale)
        cv2.line(image, (x, y), tuple(self.agent.pos), (150,150,255), 1)
        x = self.agent.pos[0] + int(np.cos((self.agent.theta+angles[1])*np.pi/180) * distances[1]*scale) 
        y = self.agent.pos[1] + int(np.sin((self.agent.theta+angles[1])*np.pi/180) * distances[1]*scale)
        cv2.line(image, (x, y), tuple(self.agent.pos), (150,150,255), 1)


    def draw_trajectory(self, image):
        for i in range(1, len(self.agent.position_history)):
            color = (255-i*5,255-i*5,255-i*5)
            cv2.line(image, self.agent.position_history[i-1], self.agent.position_history[i], color, 1)

    def to_image(self):
        array = np.frombuffer(self.bytearray, dtype=np.uint8)
        gray  = np.reshape(array, [settings.image_size, settings.image_size])
        color = cv2.applyColorMap(gray, self.colormap)
        return color

    def draw_agent(self, image):
        """
        Draws the agent in the map
        """
        cv2.circle(image, (self.agent.pos[0], self.agent.pos[1]), self.agent_radius, self.agent_color, cv2.FILLED)
        # Coordinates of the angle indicator
        x = self.agent.pos[0] + int(np.cos(self.agent.theta*np.pi/180) * 15) 
        y = self.agent.pos[1] + int(np.sin(self.agent.theta*np.pi/180) * 15)
        
        cv2.circle(self.visited, (self.agent.pos[0], self.agent.pos[1]), self.agent_radius, (2,2,2), cv2.FILLED)
        self.draw_line_at_angle(image, 15, 0, (150,150,150))
        self.draw_line_at_angle(image, 15, -135, (150,150,150))
        self.draw_line_at_angle(image, 15, 135, (150,150,150))

    
    def draw_line_at_angle(self, im, length, angle, color):
        x = self.agent.pos[0] + int(np.cos((self.agent.theta+angle)*np.pi/180) * length) 
        y = self.agent.pos[1] + int(np.sin((self.agent.theta+angle)*np.pi/180) * length)
        cv2.line(im, (x, y), tuple(self.agent.pos), (150,150,150), 1)
    
    def draw_speed(self, im=None):
        """
        Draws a small graph of the current speed
        """
        origin = self.speed_location
        off = 500
        if im is None:
            im = self.to_image()
        sp = self.agent.current_speed()
        im=cv2.rectangle(im, origin[0], (origin[0][0]+20, origin[0][1]-int(sp[0]*150)), (240, 150, 150), cv2.FILLED)
        im=cv2.rectangle(im, origin[1], (origin[1][0]+20, origin[1][1]-int(sp[1]*150)), (240, 150, 150), cv2.FILLED)
        cv2.line(im, (50,off+200), (120,off+200), (150,150,150), 1)
        cv2.line(im, (50,off+240), (120,off+240), (150,150,150), 1)
        cv2.line(im, (50,off+160), (120,off+160), (150,150,150), 1)

    def draw_elements(self, img):
        off = 500
        cv2.rectangle(img, (0, off+120), (280,off+300), (150,150,150),cv2.FILLED)
        cv2.rectangle(img, (50,off+140), (120, off+260),(255,255,255), cv2.FILLED)
        cv2.rectangle(img, (50,off+140), (120, off+260),(0,0,0), 1)
        # cv2.line(img, (0,off+120), (800,off+120), (0,0,0),1)
        cv2.putText(img, " L  R", (60,off+155), cv2.FONT_HERSHEY_PLAIN,1, (0,0,0), 1)
        # cv2.putText(img, "Simultaneous Localization and Mapping",(100,30), cv2.QT_FONT_NORMAL,1, (0,0,0), 1)
        cv2.putText(img, "0", (30,off+205), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
        cv2.putText(img, "-5", (22,off+245), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
        cv2.putText(img, "5", (30,off+165), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
        cv2.putText(img, "Step: {}".format(self.step), (150,off+150), cv2.FONT_HERSHEY_PLAIN, (1), (0,0,0),1)
        if self.wall_disp:
            cv2.putText(img, "Wall L: {}".format(self.agent.wall_l), (150,off+175), cv2.FONT_HERSHEY_PLAIN, (1), (0,0,0),1)
            cv2.putText(img, "Wall R: {}".format(self.agent.wall_r), (150,off+200), cv2.FONT_HERSHEY_PLAIN, (1), (0,0,0),1)
        

    def draw_sensor_data(self,im):
        data1 = self.agent.read_lidars()
        data = np.array([data1 for _ in range(20)])
        data *= (255.0/data.max())
        
        offsety = 600
        offsetx = 200
        for i in range(100):
            im[offsetx:offsetx+20,offsety:offsety+len(data1), 0] = data