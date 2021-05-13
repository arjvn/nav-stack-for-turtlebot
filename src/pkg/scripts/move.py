#!/usr/bin/env python
# A0158305H

from numpy import *
import rospy
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from tf2_msgs.msg import TFMessage

# ================================= CONSTANTS ==========================================
# Time Increase Constant
T_INC = 0.2

# Triangle Function Angl
HALF_ANGLE = 0.4

# PID Constants
KP_R = 0.8 # 0.8 WORKED # latest 0.3
KI_R = 0
KD_R = 0.01
KP_PHI = 0.9 # 0.9 worked # latest 0.6 # latest2: 1.2 
KI_PHI = 0 # 0.1 was working # latest 0 
KD_PHI = 0.1 # 0.1 WAS WORKING THE BEST # latest 0.2
max_vel = 0.15 # 0.22 works the best  latest: 0.2
max_ang = 0.2 # 0.7 works the best # latest 0.7


DEBUGGING = False

x_t = y_t = phi_t = x_p = y_p = 0


# def subscribe_true(msg):
#     # subscribes to the robot's true position in the simulator. This should not be used, for checking only.
#     global x_t, y_t, phi_t, x_p, y_p
#     x_t = msg.transforms[0].transform.translation.x
#     y_t = msg.transforms[0].transform.translation.y
#     phi_t = euler_from_quaternion([
#         msg.transforms[0].transform.rotation.x,
#         msg.transforms[0].transform.rotation.y,
#         msg.transforms[0].transform.rotation.z,
#         msg.transforms[0].transform.rotation.w,
#     ])[2]


# Subscribing to the odom value
def subscribe_odom(data):
    global x_t, y_t, phi_t
    x_t = data.pose.pose.position.x
    y_t = data.pose.pose.position.y
    phi_t = data.pose.pose.orientation.z


def subscribe_target(data):
    global x_p, y_p
    x_p = data.x
    y_p = data.y
    # if DEBUGGING:
    #     x_p = 2.5
    #     y_p = 0.5



###### ODOMETRY  ########


# moving to the target position from current position
def main():
    global x_t, y_t, phi_t, x_p, y_p, max_ang, max_vel

    # Node initialisation
    rospy.init_node('motion_control', anonymous=False)

    # Publisher & Subscriber
    vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
    # if not DEBUGGING:
    rospy.Subscriber("main", Odometry, subscribe_odom, queue_size=1)
    rospy.Subscriber("immediate_target", Point, subscribe_target, queue_size=1)

    # if DEBUGGING:
    #     rospy.Subscriber('tf', TFMessage, subscribe_true, queue_size=1)

    # Waiting or initialisation of robot variables maybe needed
    x_t = y_t = phi_t = nan
    x_p = y_p = nan
    it_r = it_phi = 0
    del_increase = 0

    while isnan(x_t) or isnan(x_p) and not rospy.is_shutdown():
        #print('waiting for x_t and x_p')
        pass

    # initialising variables
    pos_error = sqrt((y_p - y_t)*(y_p - y_t) + (x_p - x_t)*(x_p - x_t))
    pos_error_sum = pos_error
    prev_pos_error = pos_error
    phi_error = arctan2((y_p - y_t), (x_p - x_t)) - phi_t
    phi_error_sum = phi_error
    prev_phi_error = phi_error
    first_time = True
    v_prev = 0
    omega_prev = 0

    cmd_vel_value = Twist()
    while rospy.get_time() == 0 and not rospy.is_shutdown():
        pass
    # ---------------------------------- BEGIN ----------------------------------------------
    
    t = rospy.get_time()
    print(t)

    while (not rospy.is_shutdown()):  # required to Keyboard interrupt nicely

        if rospy.get_time() >= t:
                # initialising variables
            if not first_time: 
                pos_error = sqrt((y_p - y_t)*(y_p - y_t) + (x_p - x_t)*(x_p - x_t))
                pos_error_sum = pos_error
                prev_pos_error = pos_error
                phi_error = arctan2((y_p - y_t), (x_p - x_t)) - phi_t
                phi_error_sum = phi_error
                prev_phi_error = phi_error
                reached_target = False
            # Calculating Linear Velocity value
            pos_error = sqrt((y_p - y_t)*(y_p - y_t) + (x_p - x_t)*(x_p - x_t))
            phi_error = arctan2((y_p - y_t), (x_p - x_t)) - phi_t
            if phi_error >= pi:
                phi_error -= 2*pi
            elif phi_error < -pi:
                phi_error += 2*pi

            #integral term
            it_r = it_r + pos_error
            it_phi = it_phi + phi_error
            if abs(it_phi) > 1.0:
                it_phi = 1.0*sign(it_phi)

            #derivative term
            dt_r = pos_error - prev_pos_error
            dt_phi = phi_error - prev_phi_error

            #pid computation
            v_forward = KP_R*pos_error + KI_R*it_r + KD_R*dt_r
            omega = KP_PHI*phi_error + KI_PHI*it_phi + KD_PHI*dt_phi
            # v_forward = v_forward + 0.05 * sign(v_forward)
            # c = cos(phi_error)
            
            # v_forward = v_forward*cos(phi_error)   ##### COMMENT IF USING MOTION MODEL or NOT

            # # print 'v_forward: ', v_forward
            # print('orientation: {}'.format(phi_t))
            # # print('angle_error: {}'.format(phi_error))
            # print 'omega: ', omega
            # print'phi_error', phi_error
            # Setting up a triangle function to correct angular errors first
            if (not (phi_error > -HALF_ANGLE and phi_error < HALF_ANGLE)):               #### Uncomment this if else if using motion model
                # print('implementing half angle')
                v_forward = 0
                # del_increase = 0                                                                    
            else:
                v_forward *= (1-abs(phi_error/HALF_ANGLE))
                v_forward += 0.08 * sign(v_forward - v_prev)
                # del_increase += 0.1
                # if del_increase > 0.8: del_increase = 0.8
            if v_forward - v_prev > 0.002:
                v_forward += 0.002
            elif v_forward - v_prev < -0.002:
                v_forward -= 0.002

            if omega - omega_prev > 0.004:
                omega += 0.004
            elif omega - omega_prev < -0.004:
                omega -= 0.004
            # Setting a threshold

            #Ensure do not exceed hardware limit
            if v_forward > max_vel:
                v_forward = max_vel
            elif v_forward <-max_vel:
                v_forward = -max_vel
            if omega > max_ang:
                omega = max_ang
            elif omega <-max_ang:
                omega = -max_ang  


            if (pos_error < 0.08) and first_time: #smaller than the wheel span of the turtlebot = 287mm
                
                #run = False

                reached_target = True
                pos_error = 0
                phi_error = 0
                v_forward = 0
                omega = 0
                first_time = False
            

            # print('rbt_pos: {}'.format((x_t, y_t, phi_t)))
            # print('ep: {}'.format((x_p, y_p)))
            # print('v: {}'.format(v_forward))
            # print('w: {}'.format(omega))
            # print('ea: {}'.format(phi_error))

            # Publish the velocity values
            cmd_vel_value.linear.x = v_forward
            cmd_vel_value.angular.z = omega
            vel_pub.publish(cmd_vel_value)

            v_prev = v_forward
            omega_prev = omega

            # increment the time counter
            et = rospy.get_time() - t
            print('MOVE', et <= T_INC, et)
            t += T_INC


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass