#!/usr/bin/env python

import roslib, rospy, rospkg
from numpy import *
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Float64MultiArray
from tf.transformations import quaternion_from_euler, euler_from_quaternion, quaternion_multiply
from tf2_msgs.msg import TFMessage
import cv2
import numpy
import sys
#import lab3_aux

# ================================= CONSTANTS ==========================================        
# let's cache the SIN and POS so we don't keep recalculating it, which is slow
DEG2RAD = [i/180.0*pi for i in xrange(360)] # DEG2RAD[3] means 3 degrees in radians
SIN = [sin(DEG2RAD[i]) for i in xrange(360)] # SIN[32] means sin(32degrees)
COS = [cos(DEG2RAD[i]) for i in xrange(360)]
SQRT2 = sqrt(2)
TWOPI = 2*pi
PI = pi # numpy pi                                                      


# ================================== DATA STRUCTS ===========================================
buf = None
class OdometryMM:
    '''
        base on shorturl.at/corV3 and shorturl.at/wTV23
        
        we have these 2 formula to make a  complimentary filter
        1.filterOutput = blend*sensor1 + (1-blend)*sensor2
        blend is constant from 0 to 1, sensor 1 is odometry model , sensor 2 is imu
        2. vel = vel_prev + accel*dt 
        
        vel is sensor 2 in this case and accel is rbt_imu_a 
        3. for wt, sensor 1 is wt from odometry model, sensor 2 can just use directly as it is in w form alr.           
        dphi from slides is just dphi , while imu is the whole o
        need to add self.o to dphi in order to make use of the imu_o
        change rt to vt/wt instead of using the old formula, so lesser math operation
        using very high imu_o as odometry seems very unreliable
            
    '''
    def __init__(self, initial_pose, initial_wheels, axle_track, wheel_dia):
        self.x = initial_pose[0]
        self.y = initial_pose[1]
        self.o = initial_pose[2]
        print ("start pos is " , initial_pose)
        self.wl = initial_wheels[0]
        self.wr = initial_wheels[1]
        self.L = axle_track
        self.WR = wheel_dia / 2.0
        self.t = rospy.get_time()
        self.v = 0.0
    def calculate(self, wheels, rbt_imu_w, rbt_imu_o, rbt_imu_a):
        dt = rospy.get_time() - self.t
        dwl = wheels[0] - self.wl
        dwr = wheels[1] - self.wr
        vt = ((self.WR/ (2 * dt)) * (dwr + dwl)) * 0.5 + (self.v + rbt_imu_a * dt) * 0.5
        wt = ((self.WR/ (self.L *dt)) * (dwr - dwl)) * 0.35 + rbt_imu_w * 0.65
        dphi = self.o +  ((self.WR /self.L)* (dwr - dwl))
        dphi = ((dphi + pi) % (2*pi)) - pi
        dphi = (dphi * 0.3 + rbt_imu_o * 0.7)
        #dphi = ((dphi + pi) % (2*pi)) - pi
    
        if abs(wt) < 0.2:
            self.x =self.x +  vt * dt * cos(dphi)
            self.y = self.y +  vt * dt * sin(dphi)
        else:
            rt = vt / wt
            self.x += -rt * sin(self.o) + rt * sin(dphi)
            self.y += rt * cos(self.o) - rt * cos(dphi)
            #print(rt)
        #print(dphi, vt, wt, dt, self.o, self.x, self.y)
        self.o = dphi
            #print ("rt val", rt)
        self.t = self.t + dt
        self.wl = wheels[0]
        self.wr = wheels[1]
        self.v = vt
        #print("self . x" , self.x , self.y , self.o)
        return (self.x, self.y, self.o)
        
# ================================== SUBSCRIBERS ======================================
def subscribe_true(msg):
    # subscribes to the robot's true position in the simulator. This should not be used, for checking only.
    global rbt_true
    msg_tf = msg.transforms[0].transform
    rbt_true = (\
        msg_tf.translation.x, \
        msg_tf.translation.y, \
        euler_from_quaternion([\
            msg_tf.rotation.x, \
            msg_tf.rotation.y, \
            msg_tf.rotation.z, \
            msg_tf.rotation.w, \
        ])[2]\
    )
        
def subscribe_wheels(msg):
    global rbt_wheels
    rbt_wheels = (msg.position[1], msg.position[0])

def subscribe_imu(msg):
    global rbt_imu_o, rbt_imu_w, rbt_imu_a
    t = msg.orientation
    rbt_imu_o = euler_from_quaternion([\
        t.x,\
        t.y,\
        t.z,\
        t.w\
        ])[2]
    rbt_imu_w = msg.angular_velocity.z
    rbt_imu_a = msg.linear_acceleration.x
    
def subscribe_main(msg):
    global msg_main
    msg_main[0] = msg.data[0] # operation state
    msg_main[1] = msg.data[1] # px
    msg_main[2] = msg.data[2] # py
    
    
    
 

     
# ================================ BEGIN ===========================================
def move(start_pose):
    # ---------------------------------- INITS ----------------------------------------------
    # init node
    rospy.init_node('move')

    # Set the labels below to refer to the global namespace (i.e., global variables)
    # global is required for writing to global variables. For reading, it is not necessary
    global msg_main, rbt_imu_w, rbt_true, rbt_wheels
    
    # Initialise global vars with NaN values 
    # nan and inf are imported from numpy. If you use "import numpy as np", then nan is np.nan, and inf is np.inf.
    msg_main = [-1. for i in xrange(3)]
    rbt_true = None
    rbt_wheels = None
    rbt_imu_w = None

    # Subscribers
    rospy.Subscriber('main', Float64MultiArray, subscribe_main, queue_size=1)
    rospy.Subscriber('tf', TFMessage, subscribe_true, queue_size=1)
    rospy.Subscriber('joint_states', JointState, subscribe_wheels, queue_size=1)
    rospy.Subscriber('imu', Imu, subscribe_imu, queue_size=1)
    
    # Publishers
    publisher_u = rospy.Publisher('cmd_vel', Twist, latch=True, queue_size=1)
    publisher_move = rospy.Publisher('move', Float64MultiArray, latch=True, queue_size=1)
    # set up cmd_vel message
    u = Twist()
    # cache for faster access
    uv = u.linear #.x
    uw = u.angular #.z
    prev_v = 0
    prev_w = 0
    # set up move message
    msg_move = [0. for i in xrange(4)]
    msg_m = Float64MultiArray()
    msg_m.data = msg_move
    # publish first data for main node to register
    publisher_move.publish(msg_m)
    
    # Wait for Subscribers to receive data.
    print('[INFO] Waiting for topics... imu topic may not be broadcasting if program keeps waiting')
    while (msg_main[0] == -1. or rbt_imu_w is None or rbt_true is None or rbt_wheels is None) and not rospy.is_shutdown():
        pass

    print('[INFO] Done waiting for topics...')
    if rospy.is_shutdown():
        return
        
    # Data structures
    # ~ motion model this fuses imu and wheel sensors using a simple weighted average
    # ~ motion model also always returns an orientation that is >-PI and <PI
    # ~ notice it is intialised with true starting position and rbt_wheels
    #motion_model = lab3_aux.OdometryMM(start_pose, rbt_wheels, 0.16, 0.066) # change start_pose to rbt_true for easier debugging
    motion_model = OdometryMM(start_pose, rbt_wheels, 0.16, 0.066) # change start_pose to rbt_true for easier debugging

    err_pos = 0
    err_ang = 0
    E_pos = 0
    err_pos2 = 0 #et-1,r
    E_ang = 0 #Et0
    err_ang2 = 0 #et-1 0
    # best set of kr k0
    # 3 ,0.5, 0, 1 ,0.07 ,0.01
    # edit if found better PID. can change the ODO + IMU blend values at OdometryMM to complement it
    #kpr = 0.6
    # kir  = 0.03
    # kdr  = 0.01
    # kp0 = 0.5
    # ki0 = 0.1
    # kd0 = 0.01
    kpr = 0.6
    kir  = 0.0
    kdr  = 0.01
    kp0 = 0.55
    ki0 =0 
    kd0 = 0.01
    angle = DEG2RAD[40]

    # ---------------------------------- BEGIN ----------------------------------------------
    t = rospy.get_time()
    while (not rospy.is_shutdown()): # required to Keyboard interrupt nicely
        if (rospy.get_time() > t): # every 50 ms

            # ~ get main message
            # ~ break if main signals to stop
            if msg_main[0] == 1.:
                break
            # ~ retrieve pose (rx, ry, ro) from motion_model
            # ~ methods no longer returns tuples, but stored in object for slightly faster access
            motion_model.calculate(rbt_wheels, rbt_imu_w, rbt_imu_o, rbt_imu_a);
            rx = motion_model.x; ry = motion_model.y; ro = motion_model.o
            
            #print (rx , ry ,ro)

            # ~ publish pose to move topic
            msg_move[1] = rx; msg_move[2] = ry; msg_move[3] = ro;
            publisher_move.publish(msg_m)
            
            # calculate target coordinate to pursue
            px = msg_main[1]; py = msg_main[2]
            dx = px - rx; dy = py - ry
            # calculate positional error
            
            err_pos = sqrt(dx*dx + dy*dy)
            # calculate angular error
            err_ang = arctan2(dy, dx) - ro  #get_ang_err till elif
            if err_ang >= PI:
                err_ang -= TWOPI
            elif err_ang < -PI:
                err_ang += TWOPI
            
            
            Ptr = kpr * err_pos

            Itr = kir * E_pos
            E_pos += err_pos

            Dtr = kdr * (err_pos - err_pos2)
            
            err_pos2 = err_pos #et-1,r
            v = Ptr + Itr + Dtr
            if v > 0.20:
                v= 0.20
            elif v<-0.20:
                v=-0.20
            
            Pt0 = kp0 * err_ang
            It0 = ki0 * E_ang    

            E_ang += err_ang #Et0
            if abs(E_ang) > 1:
                E_ang = 1 * numpy.sign(E_ang)
            Dt0 = kd0*(err_ang - err_ang2)
            err_ang2 = err_ang #et-1 0
            w = Pt0 + It0 + Dt0
            if w > 0.9:
                w= 0.9
            elif w<-0.9:
                w=-0.9
            v = v * cos(err_ang) ** 3
            #print("errors are "  + str(err_ang))
            #=====================movement==================================
            #v, w = lab3_aux.get_v_w(err_pos, err_ang)                   
            uv.x = v
            uw.z = w
            publisher_u.publish(u)
            
            # increment the time counter
            et = rospy.get_time() - t
            print('[INFO] MOVE ({}, {:.3f}) v({:.3f}), w({:.3f})'.format(et <= 0.05, et, v, w))
            t += 0.05
    
    
    t += 0.3
    uv.x = 0; uw.y = 0;
    publisher_u.publish(u)
    while not rospy.is_shutdown() and rospy.get_time() < t:
        pass
        
    print('[INFO] MOVE stopped')
    
    
if __name__ == '__main__':      
    try: 
    # parse start_pose
        start_pose = sys.argv[1]
        start_pose = start_pose.split(',')
        start_pose = (float(start_pose[0]), float(start_pose[1]), 0.)

        move(start_pose)
    except rospy.ROSInterruptException:
        pass


