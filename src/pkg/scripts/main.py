#!/usr/bin/env python
# A0168924R Arjun Agrawal
import roslib, rospy, rospkg
from numpy import *
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point
from sensor_msgs.msg import LaserScan, JointState, Imu
from nav_msgs.msg import Odometry
from std_msgs import *
from tf.transformations import quaternion_from_euler, euler_from_quaternion, quaternion_multiply
from tf2_msgs.msg import TFMessage
import cv2
import numpy
import sys
# import lab2_aux
# import Project1_aux

### ARJUN'S EDITS
from pkg_turtle.msg import Motion, MsgGuider 
### ARJUN'S EDITS END

# ================================= CONSTANTS ==========================================        
# let's cache the SIN and POS so we don't keep recalculating it, which is slow
DEG2RAD = [i/180.0*pi for i in xrange(360)] # DEG2RAD[3] means 3 degrees in radians
SIN = [sin(DEG2RAD[i]) for i in xrange(360)] # SIN[32] means sin(32degrees)
COS = [cos(DEG2RAD[i]) for i in xrange(360)]
MAX_RNG = 3.5
PATH_PKG = rospkg.RosPack().get_path('pkg') + '/'
PATH_WORLDS = PATH_PKG + 'worlds/'
SQRT2 = 1.414
REL_IDX = ((1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (1,-1))
L_THRESH = 2
DELTA_VAL = 1
DEBUGGING = False
pr_vt = 0

# ================================== DATA STRUCTS ===========================================

class Cell:
    def __init__(self, idx, initial_value=0):
        """ Constructor for occupancy grid's cells
        Parameters:
            initial_value (float64):    The initial occupancy value. Default is 0
            idx (tuple of int64):       Index of cell
        """
        self.initial_value = initial_value
        self.g_cost = Distance(inf, inf)
        self.h_cost = Distance(0, 0)
        self.f_cost = Distance(inf, inf)
        self.visited = False
        self.parent = None
        self.update = 0
        self.idx = idx
        self.inf = set()
        self.occ = self.initial_value
    def reset_for_planner(self, goal_idx):
        """ Resets the cells for every run of non-dynamic path planners
        Parameters:
            goal_idx (tuple of int64):  Index of goal cell
        """
        self.g_cost = Distance(inf, inf)
        self.h_cost = Distance.from_separation(self.idx, goal_idx)
        self.f_cost = Distance(inf, inf)
        self.visited = False
        self.parent = None
        self.update = 0
    def set_occupancy(self, occupied=True):
        """ Updates the cell's observed occupancy state using the log-odds
            Binary Bayes model
        Parameters:
            occupied (bool):    If True, the cell was observed to be occupied.
                                False otherwise. Default is True
        """
 
        if (not(self.occ > 3*L_THRESH or self.occ < -3*L_THRESH)):
            if occupied:
                self.occ += DELTA_VAL
            else:
                self.occ -= DELTA_VAL
        else:
            if (self.occ > 3*L_THRESH and not occupied):
                self.occ -= DELTA_VAL
            if (self.occ < -3*L_THRESH and occupied):
                self.occ += DELTA_VAL
           
               
    #lab2_aux.set_occupancy(self, occupied)
       
    def set_inflation(self, origin_idx, add=True):
        """ Updates the cell's inflation state
        Parameters:
            origin_idx (tuple of int64):    Index of cell that is
                                            causing the current cell to
                                            be inflated / deflated
            add (bool): If True, the cell at origin_idx is newly marked
                        as occupied. If False, the cell at origin_idx
                        is newly marked as free
        """
        if add:
            self.inf.add(origin_idx)
        else:
            self.inf.discard(origin_idx)

    def is_occupied(self):
        """ Returns True if the cell is certainly occupied
        Returns:
            bool : True if cell is occupied, False if unknown or free
        """
        return self.occ > L_THRESH
    def is_inflation(self):
        """ Returns True if the cell is inflated
        Returns:
            bool : True if cell is inflated
        """
        return not not self.inf # zero length
    def is_free(self):
        """ Returns True if the cell is certainly free.
        Returns:
            bool : True if cell is free, False if unknown or occupied
        """
        return self.occ < -L_THRESH
    def is_unknown(self):
        """ Returns True if the cell's occupancy is unknown
        Returns:
            bool : True if cell's occupancy is unknown, False otherwise
        """
        return self.occ >= -L_THRESH and self.occ <= L_THRESH
    def is_planner_free(self):
        """ Returns True if the cell is traversable by path planners
        Returns:
            bool : True if cell is unknown, free and not inflated,
        """
        return self.occ <= L_THRESH and not self.inf
    def set_g_cost(self, g_cost):
        """ Sets the g-cost of the cell and recalculates the f-cost
        Parameters:
            g_cost (Distance): the Distance instance specifying the g-cost
        """
        self.g_cost = g_cost
        self.f_cost = self.g_cost + self.h_cost
    def set_h_cost(self, h_cost):
        """ Sets the h-cost of the cell and recalculates the f-cost
        Parameters:
            h_cost (Distance): the Distance instance specifying the h-cost
        """
        self.h_cost = h_cost
        self.f_cost = self.g_cost + self.h_cost
    def __str__(self):
        """ Returns the string representation of the cell,
            useful for debugging in print()
        Returns:
            str : the string containing useful information of the cell
        """
        return 'Cell{} occ:{}, f:{:6.2f}, g:{:6.2f}, h:{:6.2f}, visited:{}, parent:{}'\
        .format(self.idx, self.occ, self.f_cost.total, self.g_cost.total, self.h_cost.total, self.visited, \
        self.parent.idx if self.parent else 'None')
       
class OccupancyGrid: # Occupancy Grid
    def __init__(self, min_pos, max_pos, cell_size, initial_value, inflation_radius):
        """ Constructor for Occupancy Grid
        Parameters:
            min_pos (tuple of float64): The smallest world coordinate (x, y). This
                                        determines the lower corner of the rectangular
                                        grid
            max_pos (tuple of float64): The largest world coordinate (x, y). This
                                        determines the upper corner of the rectangular
                                        grid
            cell_size (float64): The size of the cell in the real world, in meters.
            initial_value (float64): The initial value that is assigned to all cells
            inflation_radius (float64): Inflation radius. Cells lying within the inflation
                                        radius from the center of an occupied cell will
                                        be marked as inflated
        """
        di = int64(round((max_pos[0] - min_pos[0])/cell_size))
        dj = int64(round((max_pos[1] - min_pos[1])/cell_size))

        self.cell_size = cell_size
        self.min_pos = min_pos
        self.max_pos = max_pos
        di += 1; dj += 1
        self.num_idx = (di, dj) # number of (rows, cols)
        self.cells = [[Cell((i,j), initial_value) for j in xrange(dj)] for i in xrange(di)]
        self.mask_inflation = gen_mask(cell_size, inflation_radius)
       
        # CV2 inits
        self.img_mat = full((di, dj, 3), uint8(127))
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('img', di*2, dj*2) # so each cell is 5px*5px
    def idx2pos(self, idx):
        """ Converts indices (map indices) to position (world coordinates)
        Parameters:
            idx (tuple of float64 or tuple of int64): Index tuple (i, j)
        Returns:
            tuple of float64: (i, j)
        """
        w = self.cell_size
        mp = self.min_pos
        return (idx[0] * w + mp[0], idx[1] * w + mp[1])
    def pos2idx(self, pos, rounded=True):
        """ Converts position (world coordinates) to indices (map indices)
        Parameters:
            pos (tuple of float64): Position tuple (x, y)
            rounded (bool): By default True. Set True to return a integer indices that can
                            be used to access the array. Set False to return exact indices.
        Returns:
            tuple of int64 or tuple of float64: (i, j), exact indices or integer indices
        """
        w = self.cell_size
        mp = self.min_pos
        idx = ((pos[0] - mp[0])/w, (pos[1] - mp[1])/w)
        if rounded:
            return (int64(round(idx[0])), int64(round(idx[1])))
        return idx
    def idx_in_map(self, idx): # idx must be integer
        """ Checks if the given index is within map boundaries
        Parameters:
            idx (tuple of int64): Index tuple (i, j) to be checked
        Returns:
            bool: True if in map, False if outside map
        """
        i, j = idx
        return i >= 0 and i < self.num_idx[0] and j >= 0 and j < self.num_idx[1]
    def idx2cell(self, idx, is_int=False):
        """ Retrieves the cell at a given index
        Parameters:
            idx (tuple of int64 or tuple of float64): Index tuple (i, j) of cell.
            is_int: By default False. If False, the tuple is converted to tuple of int64 by
                    rounding. Set to True to skip this only if idx is tuple of int64
        Returns:
            None or float64:    None if the idx is outside the map. float64 value of the  
                                cell if idx is in the map.
        """
        if not is_int:
            idx = (int64(round(idx[0])), int64(round(idx[1])))
        if self.idx_in_map(idx):
            return self.cells[idx[0]][idx[1]]
        return None
    def update_at_idx(self, idx, occupied):
        """ Updates the cell at the index with the observed occupancy
        Parameters:
            idx (tuple of int64): The Index of the cell to update
            occupied (bool):    If True, the cell is currently observed to be occupied.
                                False if free.
        """
        ok = self.idx_in_map
        # return if not in map
        if not ok(idx):
            return
        c = self.cells
        cell = c[idx[0]][idx[1]]
       
        # update occupancy
        was_occupied = cell.is_occupied()
        cell.set_occupancy(occupied)

        ###### ADDED ######
       
        # check if the cell occupancy state is different, and update the masks accordingly
        # (much faster than just blindly updating regardless of previous state)
        is_occupied = cell.is_occupied()
        if was_occupied != is_occupied:
            for rel_idx in self.mask_inflation:
                i = rel_idx[0] + idx[0]
                j = rel_idx[1] + idx[1]
                mask_idx = (i,j)
                if ok(mask_idx): # cell in map
                    c[i][j].set_inflation(idx, is_occupied)
                    ###### ADDED ######
    def update_at_pos(self, pos, occupied):
        """ Updates the cell at the position with the observed occupancy
        Parameters:
            pos (tuple of float64): The position of the cell to update
            occupied (bool):    If True, the cell is currently observed to be occupied.
                                False if free.
        """
        self.update_at_idx(self.pos2idx(pos), occupied)
    def show_map(self, rbt_idx, path=None, goal_idx=None, pts = None, immediate_goal = None):
        """ Prints the occupancy grid and robot position on it as a picture in a resizable
            window
        Parameters:
            rbt_pos (tuple of float64): position tuple (x, y) of robot.
        """
        c = self.cells
        img_mat = self.img_mat.copy()
        ni, nj = self.num_idx
        for i in xrange(ni):
            cc = c[i]
            for j in xrange(nj):
                cell = cc[j]
                if cell.is_occupied():
                    img_mat[i, j, :] = (255, 255, 255) # white
                elif cell.is_inflation():
                    img_mat[i, j, :] = (180, 180, 180) # light gray
                elif cell.is_free():
                    img_mat[i, j, :] = (0, 0, 0) # black
                   
        if path is not None:
            for k in xrange(len(path)):
#                idx = path[k]; next_idx = path[k+1]
                i, j = path[k]
                img_mat[i, j, :] = (0, 0, 255) # red
#                cv2.line(img_mat, idx, next_idx, (0,0,255), 1)
            #if DEBUGGING:
                
            if pts is not None:
                for point in pts:
                    i = int(point[0])
                    j = int(point[1])
                    img_mat[i, j, :] = (0, 255, 255)
                #img_mat[int(pts[-2][0]), int(pts[-2][1]), :] = (255, 0, 255)
            if immediate_goal is not None:
                goal_cell = self.pos2idx((immediate_goal.x, immediate_goal.y))
                img_mat[goal_cell[0], goal_cell[1], :] = (255, 0, 255)



           
           
        # color the robot position as a crosshair
        img_mat[rbt_idx[0], rbt_idx[1], :] = (0, 255, 0) # green
       
        if goal_idx is not None:
            img_mat[goal_idx[0], goal_idx[1], :] = (255, 0, 0) # blue

        # print to a window 'img'
  #       scale_percent = 50 # percent of original size
        # width = int(img_mat.shape[1] * scale_percent / 100)
        # height = int(img_mat.shape[0] * scale_percent / 100)
        # dim = (width, height)
        # # resize image
        # img_mat = cv2.resize(img_mat, dim, interpolation = cv2.INTER_AREA) 
        #cv2.namedWindow('img',cv2.WINDOW_NORMAL)
        cv2.imshow('img', img_mat)
        
        #cv2.resizeWindow('img', 100,100)        
        cv2.waitKey(10)
# =============================== MOTION CLASSES =========================================

class OdometryMM:
    def __init__(self, initial_pose, initial_wheels, axle_track, wheel_dia):
        self.x = initial_pose[0] # m, robot's x position in world
        self.y = initial_pose[1] # m, robot's y position in world
        self.o = initial_pose[2] # rad, robot's bearing in world
        self.wl = initial_wheels[0] # rad, robot's left angle
        self.wr = initial_wheels[1] # rad, robot's right angle
        self.L = axle_track # m, robot's axle track
        self.WR = wheel_dia/2.0 # m, robot's wheel RADIUS, not DIAMETER
        self.t = rospy.get_time() # s, time last calculated
        # print('wl wr', self.wl,self.wr)

    def calculate(self, wheels, imu_data):
        global pr_vt
        # INPUT: wheels: (left_wheel_angle, right_wheel_angle)
        # OUTPUT: a new pose (x, y, theta)
        
        # previous wheel angles stored in self.wl and self.wr, respectively. Remember to overwrite them
        # previous pose stored in self.x, self.y, self.o, respectively. Remember to overwrite them
        # previous time stored in self.t. Remember to overwrite it
        # axle track stored in self.L. Should not be overwritten.
        # wheel radius, NOT DIAMETER, stored in self.WR. Should not be overwritten.
        

        dt = rospy.get_time() - self.t # current time minus previous time
        dwl = wheels[0] - self.wl 
        dwr = wheels[1] - self.wr
        # print('wheels',wheels[0],wheels[1])
        w_imu = imu_data[0]
        ax_imu = imu_data[1]

        # fusion of IMU and wheel encoders
        w_encoder = self.WR*(dwr - dwl)/(self.L*dt)
        vt_encoder = 2*self.WR*(dwl + dwr)/(4*dt)
        vt_imu = pr_vt + ax_imu*dt


        v = 0.8*vt_encoder + 0.2*vt_imu
        w = 0.4*w_encoder + 0.6*w_imu
        # w = w_encoder
        # v = vt_encoder

        #print('v({:7.3f}) w({:7.3f}), w_enc({:7.3f}), vt_enc({:7.3f}), vt_imu({:7.3f}), ax_imu({:7.3f}), w_imu({:7.3f})'.format(v, w, w_encoder, vt_encoder, vt_imu, ax_imu, w_imu))
        # print('dt', dt)
        pr_vt = v

        # print 'imu_data: ', ax_imu, w_imu
        # print 'vt_imu: ', vt_imu, 'w_imu: ', w_imu
        # print 'vt_encoder: ', vt_encoder, 'w_encoder: ', w_encoder
        # print 'v: ', v, 'w: ', w
        #print('dwl dwr',dwl,dwr)
        deltaphi = 2*self.WR*(dwr - dwl)/(2*self.L)
        #deltaphi = w*dt
        prev_o = self.o
        self.o = (self.o + deltaphi)*0.2 + (imu_data[2])*0.8
         
        # self.o = (self.o + pi) % (2*pi) - pi
        if abs(w) < 1e-1:
        #   MM for move straight
            self.x = self.x + v*dt*cos(self.o) 
            self.y = self.y + v*dt*sin(self.o)

        else:
        #   MM for curve turns
            rt = v/w
            #print('rt', rt)
            self.x = self.x - rt*sin(prev_o) + rt*sin(self.o)
            self.y = self.y + rt*cos(prev_o) - rt*cos(self.o)
            # self.o = self.o + deltaphi

        # print(self.x, self.y, self.o)
        self.wl = wheels[0]
        self.wr = wheels[1]
        self.t = self.t + dt # update the current time. There's a reason why resampling the time is discouraged
        return (self.x, self.y, self.o)

# Define the inverse sensor model for mapping a range reading to world coordinates
def inverse_sensor_model(rng, deg, pose):
    # degree is the bearing in degrees # convert to radians
    # range is the current range data at degree
    # pose is the robot 3DOF pose, in tuple form, (x, y, o)
    x, y, o = pose
    xk = x + rng*cos(o+(deg*pi/180)) #Edited
    yk = y + rng*sin(o+(deg*pi/180)) #Edited
    

    #f.write("%.3f %.3f\n" %(xk, yk))
    
    
    return (xk, yk)

# =============================== PLANNER CLASSES =========================================  
class Distance:
    def __init__(self, ordinals=0, cardinals=0):
        """ The constructor for more robust octile distance calculation.
            Stores the exact number of ordinals and cardinals and calculates
            the approximate float64 value of the resultant cost in .total
        Parameters:
            ordinals (int64): The integer number of ordinals in the distance
            cardinals (int64): The integer number of cardinals in the distance
        """
        self.axes = [float64(ordinals), float64(cardinals)]
        self.total = SQRT2 * self.axes[0] + self.axes[1]
    def __add__(self, distance):
        """ Returns a new Distance object that represents the addition of the current
            Distance object with another Distance object. For use with '+' operator
        Parameters:
            distance (Distance): the other Distance object to add
        Returns:
            Distance : the new Distance object
        """
        return Distance(self.axes[0] + distance.axes[0], self.axes[1] + distance.axes[1])
    def __eq__(self, distance):
        """ Returns True if the current has the same number of ordinals and cardinals
            as distance. Used with '==' operator
        Parameters:
            distance (Distance): the other Distance object to equate
        Returns:
            bool : True if same number of ordinals and cardinals, False otherwise
        """
        return distance.axes[0] == self.axes[0] and distance.axes[1] == self.axes[1]
    def __ne__(self, distance):
        """ Returns True if the current has different number of ordinals and cardinals
            as distance. Used with '!=' operator
        Parameters:
            distance (Distance): the other Distance object to equate
        Returns:
            bool : True if different number of ordinals and cardinals, False otherwise
        """
        return distance.axes[0] != self.axes[0] or distance.axes[1] != self.axes[1]
    def __lt__(self, distance):
        """ Returns True if the current Distance is less than distance.
            False otherwise. Used with '<' operator
        Parameters:
            distance (Distance): the other Distance object to check
        Returns:
            bool : True if the current.total is less than distance.total
        """
        return self.total < distance.total
    def __gt__(self, distance):
        """ Returns True if the current Distance is greater than distance.
            False otherwise. Used with '>' operator
        Parameters:
            distance (Distance): the other Distance object to check
        Returns:
            bool : True if the current.total is more than distance.total
        """
        return self.total > distance.total
    def __le__(self, distance):
        """ Returns True if the current Distance is less than or equals distance.
            False otherwise. Used with '<=' operator
        Parameters:
            distance (Distance): the other Distance object to check
        Returns:
            bool :  True if current has the same number of cardinals and ordinals as
                    distance or if the current.total is less than distance.total.
                    Used with '<=' operator
        """
        return distance.axes[0] == self.axes[0] and distance.axes[1] == self.axes[1]\
            or self.total < distance.total
    def __ge__(self, distance):
        """ Returns True if the current Distance is greater than or equals distance.
            False otherwise. Used with '>=' operator
        Parameters:
            distance (Distance): the other Distance object to check
        Returns:
            bool :  True if current has the same number of cardinals and ordinals as
                    distance or if the current.total is greater than distance.total.
                    Used with '>=' operator
        """
        return distance.axes[0] == self.axes[0] and distance.axes[1] == self.axes[1]\
            or self.total > distance.total
    def __str__(self):
        """ Returns the string representation of the current Distance,
            useful for debugging in print()
        Returns:
            str : the string containing useful information of the Distance
        """
        return '${:6.2f}, O:{:6.2f}, C:{:6.2f}'.format(self.total, self.axes[0], self.axes[1])
    @staticmethod
    def from_separation(idx0, idx1):
        """ static method that returns a Distance object based on the octile distance
            between two indices idx0 and idx1
        Parameters:
            idx0 (tuple of int64): An index tuple
            idx1 (tuple of int64): Another index tuple
        """
        dj = fabs(idx0[1] - idx1[1])
        di = fabs(idx0[0] - idx1[0])
        if dj > di:
            a0 = di
            a1 = dj-di
        else:
            a0 = dj
            a1 = di-dj
        return Distance(a0, a1)
     
class Astar:
    def __init__(self, occ_grid):
        """ A* Path planner
        Parameters:
            occ_grid (OccupancyGrid) : The occupancy grid
        """
        self.occ_grid = occ_grid
    def get_path(self, start_idx, goal_idx):
        """ Returns a list of indices that represents the octile-optimal
            path between the starting index and the goal index
        Parameters:
            start_idx (tuple of int64): the starting index
            goal_idx (tuple of int64): the goal index
        Returns:
            list of tuple of int64: contains the indices in the optimal path
        """
        goal_found = False
        occ_grid = self.occ_grid
        open_list = OpenList()

        # get number of rows ni (x) and number of columns nj (y)
        ni, nj = occ_grid.num_idx
        path = []

        #print('Map rows, columns: {}'.format((ni, nj)))
       
        # resets h-cost, g-cost, update and occ for all cells
        for i in xrange(ni):
            for j in xrange(nj):
                occ_grid.idx2cell((i,j)).reset_for_planner(goal_idx)
                # !(settled) use occ_grid.idx2cell() and the cell's reset_for_planner()
                # pass
               
        # put start cell into open list
       
        start_cell = occ_grid.idx2cell(start_idx)
        start_cell.set_g_cost(Distance(0, 0))
        open_list.add(start_cell)
        #print('start_cell: ')
        #print(start_cell)
       
        # !(settled) get the start cell from start_idx
        # !(settled) set the start cell distance using set_g_cost and Distance(0, 0)
        # !(settled) add the cell to open_list
        # m = 0
        # now we non-recursively search the map
        while open_list.not_empty():
            # m += 1
            # if m == 1000: 
            #     print(open_list)
            #     raise Exception()
            #print(open_list)
            #print("bp0")
            cell = open_list.remove()
            #print (cell)
            # skip if already visited, bcos a cheaper path was already found
            if cell.visited:            
                continue
           
            cell.visited = True 
            #print("bp1")  
            # !(settled) set the cell as visiteds
           
            if goal_found:
                cell = occ_grid.idx2cell(goal_idx)

            # goal
            if cell.idx == goal_idx:              
                while True:
                    path.append(cell.idx)
                    #print("bp1.5")
                    cell = cell.parent
                    #print(cell)
                    if cell == None:
                        #print("bp2")
                        break
                    # !(settled) append the cell.idx onto path
                    # !(settled) let cell = cell's parent
                    # !(settled) if cell is None, break out of the while loop
                    #pass
                break # breaks out of the loop: while open_list.not_empty()
               
            # if not goal or not visited, we try to add free neighbour cells into the open list
            #print("bp2.5")
            for nb_cell in self.get_free_neighbors(cell):
                change_g = Distance.from_separation(cell.idx, nb_cell.idx)
                

                if nb_cell.is_inflation():
                    change_g = Distance(change_g.axes[0]*30, change_g.axes[1]*30)
                    
                tent_g_cost = cell.g_cost + change_g
                #print("bp3")
                #print (cell)
                if tent_g_cost < nb_cell.g_cost:
                    nb_cell.set_g_cost(tent_g_cost)
                    nb_cell.parent = cell
                    open_list.add(nb_cell)
                    #print("bp4")
                idx = nb_cell.idx; ni = idx[0]; nj = idx[1]
                if (ni == goal_idx[0] and nj == goal_idx[1]):
                    goal_found = True
                    break
                # !(unsettled) calculate the tentative g cost of getting from current cell (cell) to neighbouring cell (nb_cell)...
                # !(unsettled)     use cell.g_cost and Distance.from_separation()
                # !(settled) if the tentative g cost is less than the nb_cell.g_cost, ...
                # !(settled)     1. assign the tentative g cost to nb_cell's g cost using set_g_cost
                # !(settled)     2. set the nb_cell parent as cell
                # !(settled)     3. add the nb_cell to the open list using open_list.add()
                #pass
                   
        return path
           
    def get_free_neighbors(self, cell):
        """ Checks which of the 8 neighboring cells around a cell are in the map,
            free, unknown and not inflated and returns them as a list
        Parameters:
            cell (Cell): the cell in the occupancy grid
        Returns:
            list of Cells: the list of neighboring cells which are in the map,
            free, unknown and not inflated
        """
        # start from +x (N), counter clockwise
        occ_grid = self.occ_grid
        neighbors = []
        idx = cell.idx
        for rel_idx in REL_IDX:
            nb_idx = (rel_idx[0] + idx[0], rel_idx[1] + idx[1]) #non-numpy
            nb_cell = occ_grid.idx2cell(nb_idx)
            if nb_cell is not None: # and nb_cell.is_planner_free() add even the inflated cells
                neighbors.append(nb_cell)
        return neighbors

       
class OpenList:
    def __init__(self):
        """ The constructor for the open list
        """
        # initialise with an list (array)
        self.l = []
    def add(self, cell):
        """ Adds the cell and sorts it based on its f-cost followed by the h-cost
        Parameters:
            cell (Cell): the Cell to be sorted, updated with f-cost and h-cost information
        """
        # set l as the open list
        l = self.l
       
        # if l is empty, just append and return
        if not l:
            l.append(cell)
            return
       
        # now we sort and add
        i = 0; nl = len(l)
        # we start searching from index (i) 0, where the cells should be cheapest
        while i < nl:
            list_cell = l[i]
            if cell.f_cost < list_cell.f_cost or (cell.f_cost == list_cell.f_cost and cell.h_cost < list_cell.h_cost):
                #print("bp5")
                break
            # !(settled) get the cell (list_cell) in the index (i) of the open list (l)
            # !(settled) now if the cell's f_cost is less than the list_cell's f_cost, ...
            # !(settled)     or if the cell's f_cost = list_cell's f_cost but ...
            # !(settled)     cell's h_cost is less than the list_cell's h_cost...
            # !(settled)     we break the loop (while i < nl)
           
            # increment the index
            i += 1
           
        # insert the cell into position i of l
        l.insert(i, cell)
    def remove(self):
        """ Removes and return the cheapest cost cell in the open list
        Returns:
            Cell: the cell with the cheapest f-cost followed by h-cost
        """
        #temp_cell = self.l[0]
        #print(self.l[0])
        #self.l.remove(self.l[0])
        #print(self.l)
        #print("bp6")
        return self.l.pop(0)
        # return the first element in self.l
        #print("returning nothing?")
        #pass
    def not_empty(self):
        #print("surpassed remove")
        return not not self.l # self.l is False if len(self.l) is zero ==> faster
    def __str__(self):
        l = self.l
        s = ''
        for cell in l:
            s += '({:3d},{:3d}), F:{:6.2f}, G:{:6.2f}, H:{:6.2f}\n'.format(\
                cell.idx[0], cell.idx[1], cell.f_cost.total, cell.g_cost.total, cell.h_cost.total)
        return s
   

class GeneralLOS:
    def __init__(self, map):
        self.pos2idx = map.pos2idx # based on the map (occ_grid) it returns the map index representation of the position pos
        # use self.pos2idx(pos, False) to return the exact index representation, including values that are less than 1.
        # use self.pos2idx(pos) to return the integer index representation, which is the rounded version of self.pos2idx(pos, False)
    def calculate(self, start_pos, end_pos):
        # sets up the LOS object to prepare return a list of indices on the map starting from start_pos (world coordinates) to end_pos (world)
        # start_pos is the robot position.
        # end_pos is the matomum range of the LIDAR, or an obstacle.
        # every index returned in the indices will be the index of a FREE cell
        # you can return the indices, or update the cells in here
        start_idx = self.pos2idx(start_pos)
        end_idx = self.pos2idx(end_pos)
        indices = [] # init an empty list

        jf, to, tf, jo = end_idx[0], start_idx[0], end_idx[1], start_idx[1]  
        # Get differences
        di = jf - to
        dj = tf - jo
        # Assign short and long
        if (abs(di) > abs(dj)):
            dl = di
            ds = dj
            l, s = to , jo
            lf, sf = jf, tf
            get_idx = lambda l, s : (l, s)
        else:
            dl = dj
            ds = di
            l, s = jo, to
            lf, sf = tf, jf
            get_idx = lambda l, s : (s, l)
        
        # Get signs and increments
        del_s = sign(ds)
        del_l = sign(dl)
        psi_sl = 2*ds
        eta_sl = 2*abs(dl)*del_s
        
        # Get Error
        error_sl = 0
        # Get lambda
        lambda_sl = abs(ds)-abs(dl)
        # Get Error Checker
        
        if ds >= 0:
            has_big_err = lambda error_ksl : error_ksl >= abs(dl)
        else:
            has_big_err = lambda error_ksl : error_ksl < -abs(dl)
        # Propagate
        while (l, s) != (lf, sf):
            l += del_l
            error_sl += psi_sl
            if has_big_err(error_sl):
                error_sl -= eta_sl
                s += del_s
                # Previous cells
                lambdabar_sl = error_sl*del_s
                if lambdabar_sl < lambda_sl:
                    idx = get_idx(l, s-del_s)
                    indices.append(idx)
                    if (l, s-del_s) == (lf, sf): break
                elif lambdabar_sl > lambda_sl:
                    idx = get_idx(l-del_l, s)
                    indices.append(idx)
                    if (l-del_l, s) == (lf, sf): break

            idx = get_idx(l, s)
            indices.append(idx)
        return indices

def generalLOS(start_pos, end_pos, indexed, occ_grid):
    # sets up the LOS object to prepare return a list of indices on the map starting from start_pos (world coordinates) to end_pos (world)
    # start_pos is the robot position.
    # end_pos is the maximum range of the LIDAR, or an obstacle.
    # every index returned in the indices will be the index of a FREE cell
    # you can return the indices, or update the cells in here
    if indexed:
        start_idx = int(round(start_pos[0])), int(round(start_pos[1]))
        end_idx = int(round(end_pos[0])), int(round(end_pos[1]))
        start_flt = start_pos
        end_flt = end_pos
        # print(start_flt, end_flt)
    else:
        start_idx = occ_grid.pos2idx(start_pos)
        end_idx = occ_grid.pos2idx(end_pos)
        start_flt = occ_grid.pos2idx(start_pos, False)
        end_flt = occ_grid.pos2idx(end_pos, False)
        # print('-----', start_idx, end_idx, start_flt, end_flt)

    indices = [] # init an empty list

    xef, xei, yef, yei = end_flt[0], start_flt[0], end_flt[1], start_flt[1] 
    xf, xi, yf, yi =  end_idx[0], start_idx[0], end_idx[1], start_idx[1] 
    # print('ssss', indexed, (xi, yi), (xf, yf), (xei, yei), (xef, yef))

    # Get differences
    Dx = xef - xei
    Dy = yef - yei
    # Assign short and long
    if (abs(Dx) >= abs(Dy)):
        Dl = Dx
        Ds = Dy
        get_idx = lambda l, s : (l, s)
    else:
        Dl = Dy
        Ds = Dx
        get_idx = lambda l, s : (s, l)

    le, se = get_idx(xei,yei)
    lef, sef = get_idx(xef,yef)
    l, s = get_idx(xi, yi)
    lf, sf = get_idx(xf, yf)
    # print((le,se), (l, s))

    # Get signs and increments
    dels = int(sign(Ds))
    dell = int(sign(Dl))
    psi_s = Ds/abs(Dl)

    # print('klklklk', psi_s, Dl, Ds)
    # eta_sl = 2*abs(Dl)*dels
    # Get Error
    epsilon_s = se - s
    epsilon_l = le - l 
    # print(epsilon_s, epsilon_l)
    # Get lambda
    lambda_sl = abs(Ds/Dl)*(0.5 + epsilon_l*dell)-0.5# abs(Ds)-abs(Dl)
    # Get Error Checker
    if dels >= 0:
        has_big_err = lambda epsilon_ksl : epsilon_ksl >= 0.5 
    else:
        has_big_err = lambda epsilon_ksl : epsilon_ksl < -0.5
    # Propagate
    i = 0
    while l != lf or s != sf:
        i += 1
        #if i == 100: raise Exception()

        l += dell
        epsilon_s += psi_s
        # print((l,s), (lf, sf))
        if has_big_err(epsilon_s):
            epsilon_s -= dels #eta_sl
            s += dels
            # Previous cells
            lambdabar_sl = epsilon_s*dels
            if lambdabar_sl < lambda_sl:
                idx = get_idx(l, s-dels)
                indices.append(idx)
                if l == lf and s-dels == sf: break
            elif lambdabar_sl > lambda_sl:
                idx = get_idx(l-dell, s)
                indices.append(idx)
                if l-dell == lf and s == sf: break
            # else:
                # idx = get_idx(l-dell, s)
                # indices.append(idx)
                # if (l-dell, s) == (lf, sf): break
                # idx = get_idx(l, s-dels)
                # indices.append(idx)
                # if (l, s-dels) == (lf, sf): break
        idx = get_idx(l, s)
        # print(idx, (lf, sf))
        indices.append(idx)
    return indices
        
def generalLOSInt(start_idx, end_idx):
    xf, xi, yf, yi = end_idx[0], start_idx[0], end_idx[1], start_idx[1]  
    # Get differences
    Dx = xf - xi
    Dy = yf - yi
    # Assign short and long
    if (abs(Dx) > abs(Dy)):
        Dl = Dx
        Ds = Dy
        l, s = xi , yi
        lf, sf = xf, yf
        get_idx = lambda l, s : (l, s)
    else:
        Dl = Dy
        Ds = Dx
        l, s = yi, xi
        lf, sf = yf, xf
        get_idx = lambda l, s : (s, l)
    # Get signs and increments
    dels = sign(Ds)
    dell = sign(Dl)
    psi_sl = 2*Ds
    eta_sl = 2*abs(Dl)*dels
    # Get Error
    epsilon_sl = 0
    # Get lambda
    lambda_sl = abs(Ds)-abs(Dl)
    # Get Error Checker
    if Ds >= 0:
        has_big_err = lambda epsilon_ksl : epsilon_ksl >= abs(Dl)
    else:
        has_big_err = lambda epsilon_ksl : epsilon_ksl < -abs(Dl)
    # Propagate
    indices = [get_idx(l, s)]
    while (l, s) != (lf, sf):
        l += dell
        epsilon_sl += psi_sl
        if has_big_err(epsilon_sl):
            epsilon_sl -= eta_sl
            s += dels
            # Previous cells
            lambdabar_sl = epsilon_sl*dels
            if lambdabar_sl < lambda_sl:
                idx = get_idx(l, s-dels)
                indices.append(idx)
                if (l, s-dels) == (lf, sf): break
            elif lambdabar_sl > lambda_sl:
                idx = get_idx(l-dell, s)
                indices.append(idx)
                if (l-dell, s) == (lf, sf): break
        idx = get_idx(l, s)
        indices.append(idx)
    return indices


def segment(astar_path, occ_grid): # only for A* path
    segs = [] # list of segments
    # idx = [astar_path[0]] # container for single segment  
    # was_inf = occ_grid.idx2cell((idx[0][0], idx[0][1])).is_inflation()# status of each segment
    idx = astar_path[0] # container for single segment  
    was_inf = occ_grid.idx2cell((idx[0], idx[1])).is_inflation()# status of each segment
    seg = [idx]
    segs_inf = [was_inf]
    for a in range(1,len(astar_path)):
        idx = astar_path[a]
        is_inf = occ_grid.idx2cell((idx[0], idx[1])).is_inflation()
        if was_inf != is_inf:
            segs.append(seg)
            seg = [idx]
            segs_inf.append(is_inf)
        else:
            seg.append(idx)
        was_inf = is_inf
    # confusion here
    segs.append(seg)

    return segs, segs_inf

def turn_points(segs): # only for A* path, extract turning points regardless of inflation status
    segs_pts = []
    seg_pts = []
    for seg in segs:
        # start and end always appended hence
        seg_pts = [0]
        seg_len = len(seg)
        if seg_len == 1:
            segs_pts.append(seg_pts)
            continue
        # elif seg_len == 2:
        #     seg_pts = [0,1]
        #     segs_pts.append(seg_tp)
        elif seg_len > 2:
            i0,j0 = seg[1]
            i1,j1 = seg[0]
            di0 = i0 - i1 # do not pack to tuple to speed up
            dj0 = j0 - j1
            for n in range(2,seg_len):
                i1,j1 = seg[n]
                di1 = i1 - i0
                dj1 = j1 - j0
                if di0 != di1 or dj0 != dj1:
                    seg_pts.append(n-1)
                di0 = di1
                dj0 = dj1
                i0 = i1
                j0 = j1
        seg_pts.append(seg_len - 1)
        segs_pts.append(seg_pts) 

    return segs_pts    

def los_points(occ_grid, segs, segs_pts, segs_inf, forward):
# initialise based on forward
    if forward: # goal to start
        get_first = lambda length : 0 # first turn pt. (start pt. in seg), does not depend on length
        get_third = lambda length : 2 # third turn pt., does not depend on length
        get_last = lambda length: length - 1 # last turn pt. (last pt. in seg)
        next_p = lambda p, direct_LOS : (p + 2 if direct_LOS else p + 1) # get next turn pt.
        next_q = lambda q : q - 1 # get next pt. in seg if there are obs in LOS
        at_end = lambda p, length : p == length # checks if p is at the end
        aft_end = lambda p, length : p > length # checks if p is far from the end
    else: # start to goal
        get_first = lambda length : length - 1
        get_third = lambda length : length - 3
        get_last = lambda length : 0 # does not depend on length
        next_p = lambda p, direct_LOS : (p - 2 if direct_LOS else p - 1)
        next_q = lambda q : q + 1
        at_end = lambda p, length : p == -1 # does not depend on length should be used.
        aft_end = lambda p, length : p < -1 # does not depend on length
    # initialise new list to hold all LOS pts in all segments

    new_segs_pts = []
    # for every segment
    for a in range(len(segs)):
    # skip if segment is inflated
        if segs_inf[a]:
            new_segs_pts.append(False)
            continue
        # initialise new segment for the LOS pts in the current segment
        seg_pts = segs_pts[a]
        seg_pts_len = len(seg_pts)
        p = get_first(seg_pts_len) # 0 if forward, seg_pts_len - 1 if backward
        new_seg_pts = [seg_pts[p]]
        if seg_pts_len == 2:
            p = get_last(seg_pts_len) # seg_pts_len - 1 if forward, 0 if backward
            new_seg_pts.append(seg_pts[p])
            new_segs_pts.append(new_seg_pts)
        elif seg_pts_len < 2:
            new_segs_pts.append(new_seg_pts)
        else: # seg_pts_len > 2
            seg = segs[a]
            p = get_first(seg_pts_len) # first turning pt, actually just the first in seg
            start_idx = seg[seg_pts[p]] # get the idx of the first turning pt
            # get third turning pt
            p = get_third(seg_pts_len) # 2 for forward, seg_pts_len - 3 for backward
            q = seg_pts[p]
            end_idx = seg[q] # get the idx of the third turning pt.
            # set LOS
            direct_LOS = True
            los_indices = generalLOSInt(start_idx, end_idx)
            l = 0
            while True:
                l += 1
                idx = los_indices[l]
                if idx == end_idx:
                    new_seg_pts.append(q)
                    p = next_p(p, direct_LOS) # get the next turning point in each segment
                    if at_end(p, seg_pts_len): # == seg_pts_len for forward, == -1 for backward
                        p = get_last(seg_pts_len)
                        new_seg_pts.append(seg_pts[p])
                        break
                    elif aft_end(p, seg_pts_len): # > seg_pts_len for forward, < -1 for backward
                        break
                    else: # some turn points left in segment
                        direct_LOS = True
                        start_idx = end_idx
                        q = seg_pts[p] # get the next turn pt.
                        end_idx = seg[q] # get the next turn pt. index
                        los_indices = generalLOSInt(start_idx, end_idx)
                        l = 0
                        continue
                    # check if los crosses an inflated cell
                if occ_grid.idx2cell((idx[0], idx[1])).is_inflation():
                    direct_LOS = False
                    # set the end idx one cell backward in seg
                    q = next_q(q) # q-1 for forward, q+1 for backward
                    end_idx = seg[q]
                    los_indices = generalLOSInt(start_idx, end_idx)
                    l = 0
                    # same indent level as comment: # skip if segment is inflated
            new_segs_pts.append(new_seg_pts)
    return new_segs_pts

def processed_points(segs, segs_inf, segs_pts, g_segs_pts, s_segs_pts):
    pts = []
    # for every segment
    for a in range(len(segs)):
    # preserve turning points on inflated segments
        seg = segs[a]
        seg_pts = segs_pts[a]
        if segs_inf[a]:
            for q in segs_pts[a]:
                pts.append((float(seg[q][0]), float(seg[q][1])))
            continue
        
        g_seg_pts = g_segs_pts[a]
        s_seg_pts = s_segs_pts[a]
        seg_pts_len = len(g_seg_pts)
        p = 0; q = seg_pts_len - 1 # s_seg_pts and g_seg_pts length are the same
        while p < seg_pts_len:
            g = g_seg_pts[p]
            s = s_seg_pts[q]
            if g == s:
                pts.append((float(seg[g][0]), float(seg[g][1]))) # convert the int index to float index
                p += 1
                q -= 1
                continue
            # Else
            g_is_elbow = g in seg_pts # ! 20200227
            s_is_elbow = s in seg_pts # ! 20200227
            if g_is_elbow and not s_is_elbow: # ! 20200227
                pts.append(seg[s]) # ! 20200227
            elif s_is_elbow and not g_is_elbow: # ! 20200227
                pts.append(seg[g]) # ! 20200227
            elif not g_is_elbow and not s_is_elbow: # ! 20200227
                i0, j0 = seg[g_seg_pts[p - 1]]
                i1, j1 = seg[g]
                i2, j2 = seg[s_seg_pts[q - 1]]
                i3, j3 = seg[s]
                a0 = j0 - j1; a1 = j2 - j3
                b0 = i1 - i0; b1 = i3 - i2
                c0 = a0*i0 + b0*j0
                c1 = a1*i2 + b1*j2
                d = float(a0*b1 - b0*a1)
                idx = ((b1*c0 - b0*c1)/d, (a0*c1 - a1*c0)/d) # float form
                pts.append(idx)
            p += 1
            q -= 1
    return pts

def processed_path(pts, occ_grid):
    start_idx = pts[0] # pts should be in float form
    path = [(int(start_idx[0]), int(start_idx[1]))] # assuming los does not return the start_idx
    # for every segment
    for p in range(1, len(pts)):
        end_idx = pts[p]
        for idx in generalLOS(start_idx, end_idx, True, occ_grid):
        # assuming los does not return the start_idx
            path.append(idx)
        start_idx = end_idx
    return path

def post_process(astar_path, occ_grid):
    if len(astar_path) == 2:
        return [astar_path[0], astar_path[1]], [astar_path[0], astar_path[1]]
    elif len(astar_path) == 1:
        return [a_path[0]], [a_path[0]]

    segs, segs_inf = segment(astar_path, occ_grid)
    # print('segs {}'.format(segs))
    # print('segs_inf {}'.format(segs_inf))
    segs_pts = turn_points(segs)
    # print('segs_pts {}'.format(segs_pts))
    g_segs_pts = los_points(occ_grid, segs, segs_pts, segs_inf, True)
    # print('g_segs_pts {}'.format(g_segs_pts))
    s_segs_pts = los_points(occ_grid, segs, segs_pts, segs_inf, False)
    # print('s_segs_pts {}'.format(s_segs_pts))
    pts = processed_points(segs, segs_inf, segs_pts, g_segs_pts, s_segs_pts)
    # print('pts {}'.format(pts))
    path = processed_path(pts, occ_grid)

    return path, pts

# for every segment
# =============================== SUBSCRIBERS =========================================  
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
   
def subscribe_scan(msg):
    # stores a 360 long tuple of LIDAR Range data into global variable rbt_scan.
    # 0 deg facing forward. anticlockwise from top.
    global rbt_scan, write_scan, read_scan
    write_scan = True # acquire lock
    if read_scan:
        write_scan = False # release lock
        return
    rbt_scan = msg.ranges
    write_scan = False # release lock
def subscribe_wheels(msg):
    # returns the angles in which the wheels have been recorded to turn since the start
    global rbt_wheels
    right_wheel_angle = msg.position[0] # examine topic /joint_states #Edited
    left_wheel_angle = msg.position[1] # examine topic /joint_states #Edited
    rbt_wheels = (left_wheel_angle, right_wheel_angle)
    return rbt_wheels

    # global rbt_wheels
    # rbt_wheels = Project1_aux.subscribe_wheels(msg)
   
def get_scan():
    # returns scan data after acquiring a lock on the scan data to make sure it is not overwritten by the subscribe_scan handler while using it.
    global write_scan, read_scan
    read_scan = True # lock
    while write_scan:
        pass
    scan = rbt_scan # create a copy of the tuple
    read_scan = False
    return scan
   
def subscribe_imu(msg):
    global InertialMU
    w_imu = msg.angular_velocity.z #angular_vel z
    ax_imu = msg.linear_acceleration.x #lin_vel x
    #ay_imu = msg.linear_acceleration.y #lin_vel y
    imu_orn_msg = msg.orientation
    imu_orn = euler_from_quaternion([
        imu_orn_msg.x,
        imu_orn_msg.y,
        imu_orn_msg.z,
        imu_orn_msg.w
    ])[2]
    InertialMU = (w_imu, ax_imu, imu_orn)
# ================================== PUBLISHERS ========================================

### ARJUN'S EDITS
pub_motion_turtle = rospy.Publisher('/turtle/motion', Motion, latch=True, queue_size=1)
msg_motion_turtle = Motion()
pub_motion_turtle.publish(msg_motion_turtle)

pub_guider_turtle = rospy.Publisher('/turtle/guider', MsgGuider, latch=True, queue_size=1)
msg_guider_turtle = MsgGuider()
pub_guider_turtle.publish(msg_guider_turtle)
### ARJUN'S EDITS END

# =================================== OTHER METHODS =====================================
def gen_mask(cell_size, radius):
    """ Generates the list of relative indices of neighboring cells which lie within
        the specified radius from a center cell
    Parameters:
        radius (float64): the radius
    """
    radius_mapped = radius/cell_size
    index_masking = int64(radius_mapped)
   
    indices = []
   
    j = 0
    while float(j) < radius_mapped:
        i_max = int64(round(sqrt(radius_mapped*radius_mapped-j*j)))
        for i in range(-i_max,i_max+1):
            indices.append([i,j])
            indices.append([i,-j])
        j = j + 1
   
    return indices

# ================================ BEGIN ===========================================
MotionModel = OdometryMM
LOS = GeneralLOS
#LOS = lab2_aux.GeneralLOS
PathPlanner = Astar #lab2_aux.Astar
def main(goals, cell_size, min_pos, max_pos):
# def main():
    # ---------------------------------- INITS ----------------------------------------------
    # init node
    rospy.init_node('main')
   
    # Set the labels below to refer to the global namespace (i.e., global variables)
    # global is required for writing to global variables. For reading, it is not necessary
    global rbt_scan, rbt_true, read_scan, write_scan, rbt_wheels, rbt_control, InertialMU
   
    # Initialise global vars with NaN values
    # nan and inf are imported from numpy. If you use "import numpy as np", then nan is np.nan, and inf is np.inf.
    rbt_scan = [nan]*360 # a list of 360 nans
    rbt_true = [nan]*3
    rbt_wheels = [nan]*2
    read_scan = False
    write_scan = False
    InertialMU = [nan]*3
    pub_count = 0
    goal_x = 0
    goal_y = 0
    goal_reached = False
    path_counter = 0

    # Subscribers
    rospy.Subscriber('scan', LaserScan, subscribe_scan, queue_size=1)
    rospy.Subscriber('tf', TFMessage, subscribe_true, queue_size=1)
    rospy.Subscriber('joint_states', JointState, subscribe_wheels, queue_size=1)
    rospy.Subscriber('imu', Imu, subscribe_imu, queue_size=1) #imu

    # Publishers
    goal_pub = rospy.Publisher("immediate_target", Point, queue_size=10)
    immediate_goal = Point()
    rbt_pos_pub = rospy.Publisher("main", Odometry, queue_size=10)
    curr_rbt_pos = Odometry()
   
    #print InertialMU
    # Wait for Subscribers to receive data.
    while isnan(rbt_scan[0]) or isnan(rbt_true[0]) or isnan(rbt_wheels[0]) or isnan(InertialMU[0]):
        pass
   
    # Data structures
    occ_grid =  OccupancyGrid(min_pos, max_pos, cell_size, 0, 0.22)#OccupancyGrid((-10,-10), (4,4), 0.1, 0, 0.3) # #OccupancyGrid((-4,-4), (4,4), 0.1, 0, 0.2)#world1: OccupancyGrid((-10,-10), (4,4), 0.1, 0, 0.3) # OccupancyGrid((-4,-4), (4,4), 0.1, 0, 0.2)
    los = LOS(occ_grid)

    if DEBUGGING:
        motion_model = None #MotionModel(rbt_true, rbt_wheels, 0.16, 0.066)
    else:
        motion_model = MotionModel(rbt_true, rbt_wheels, 0.16, 0.066)
    planner = PathPlanner(occ_grid)
    goal_number = 0
    goal_pos = goals[goal_number]#(-8.5, -6)#(-5, -1)#(-8.5, -6)# (1.5, 0.5)#world1: (-8.5, -6)
    # goal_pos = (0.5, 0.5)
    
   
    # ---------------------------------- BEGIN ----------------------------------------------
    t = rospy.get_time()
    update = 0
    while (not rospy.is_shutdown()): # required to Keyboard interrupt nicely
       
        if (rospy.get_time() > t): # every 50 ms
           
            # get scan
            scan = get_scan()
           
            # calculate the robot position using the motion model
            if DEBUGGING:
                rbt_pos = rbt_true
            else:
                rbt_pos = motion_model.calculate(rbt_wheels, InertialMU)
            curr_rbt_pos.pose.pose.position.x = rbt_pos[0]
            curr_rbt_pos.pose.pose.position.y = rbt_pos[1]
            curr_rbt_pos.pose.pose.orientation.z = rbt_pos[2]
            rbt_pos_pub.publish(curr_rbt_pos)

### ARJUN'S EDITS
            msg_motion_turtle.x = rbt_pos[0]
            msg_motion_turtle.y = rbt_pos[1]
            msg_motion_turtle.o = rbt_pos[2]
            pub_motion_turtle.publish(msg_motion_turtle)
### NOTE: VELOCITY AND ANGULAR VELOCITY VALUES NOT PUBLISHED SINCE PREDICTION METHOD IN HECTOR_GUIDER DUE CHANGE - REFER TO H_GUIDER LINE: 102
### ### ARJUN'S EDITS END

            #print('rbt_pose: {}'.format(rbt_pos))
            #print (rbt_pos) #rbt_pos is in meters
           
            # increment update iteration number
            update += 1
           
            # for each degree in the scan
            for i in xrange(360):
                if scan[i] != inf: # range`` reading is < max range ==> occupied
                    end_pos = inverse_sensor_model(scan[i], i, rbt_pos)
                    # set the obstacle cell as occupied
                    occ_grid.update_at_pos(end_pos, True)
                else: # range reading is inf ==> no obstacle found
                    end_pos = inverse_sensor_model(MAX_RNG, i, rbt_pos)
                    # set the last cell as free
                    occ_grid.update_at_pos(end_pos, False)
                # set all cells between current cell and last cell as free
                l = los.calculate(rbt_pos, end_pos)
                if len(l) != 0:
                    l.pop()
                for idx in l:
                    occ_grid.update_at_idx(idx, False)
           
            # plan
            rbt_idx = occ_grid.pos2idx(rbt_pos)
            #print('rbt_idx: {}'.format(rbt_idx))
            goal_idx = occ_grid.pos2idx(goal_pos)
            #print('goal_idx: {}'.format(goal_idx))
            # print('b4 A*')
            path = planner.get_path(rbt_idx, goal_idx)

            #if(len(path))!=0 and path_counter < 5:
                # implementing post processing here
            # print('finish A*')
            path, pts = post_process(path, occ_grid)
            # print('finish post_process')
                # path_counter += 1
            #print('path: {}'.format(path))

            # if not DEBUGGING: 
            #     connections = goal_pub.get_num_connections()
            #     #print('connections: {}'.format(connections))
            #     if (path is not None and (rbt_idx[0] + 3 > goal_x and rbt_idx[0] - 3 < goal_x\
            #                 and rbt_idx[1] + 3 > goal_y and rbt_idx[1] - 3 < goal_y))\
            #                     or (connections < 1) or pub_count > 3: #not very close or 0 connection # and pub_count > 6:

            #         if pts is not None:
            #             # if path[len(path)][0]*occ_grid.cell_size - immediate_goal.x > 2
            #             goal_x = int(pts[-2][0])
            #             goal_y = int(pts[-2][1])

            #             # See if the robot is already very close
            #             if rbt_idx[0] + 3 > goal_x and rbt_idx[0] - 3 < goal_x\
            #                 and rbt_idx[1] + 3 > goal_y and rbt_idx[1] - 3 < goal_y\
            #                     and len(pts) > 2:
            #                 print('Given a goal further ahead')
            #                 goal_x = int(pts[-3][0])
            #                 goal_y = int(pts[-3][1])                        

            #             print ('curr goal idx: {}, {}'.format(goal_x, goal_y))
            #         else:

            #             goal_x = path[0][0]
            #             goal_y = path[0][1]
            #             print ('final goal: {}, {}'.format(goal_x, goal_y)) 
            
            #         #pts_len = len(pts)
            #         immediate_goal.x = goal_x*occ_grid.cell_size + occ_grid.min_pos[0]
            #         immediate_goal.y = goal_y*occ_grid.cell_size + occ_grid.min_pos[1]
            #         goal_pub.publish(immediate_goal)
            #         pub_count = 0
            #     else:
            #         pub_count += 1

            # else:
            if path is not None: #not very close or 0 connection # and pub_count > 6:
                if pts is not None:
                    # if path[len(path)][0]*occ_grid.cell_size - immediate_goal.x > 2
                    goal_x = int(pts[-2][0])
                    goal_y = int(pts[-2][1])

                    g = -3
                    # See if the robot is already very close
                    if abs(goal_x - rbt_idx[0]) < 3 and abs(goal_y - rbt_idx[1]) < 3 and g >= -len(pts):
                        #print('Given a goal further ahead'
                        goal_x = int(pts[g][0])
                        goal_y = int(pts[g][1])     
                        
                        # print ('help3')                   

                    #print ('curr goal idx: {}, {}'.format(goal_x, goal_y))
                else:
                    goal_x = path[0][0]
                    goal_y = path[0][1]
                    #print ('final goal idx: {}, {}'.format(goal_x, goal_y))
                    # print ('help4') 
            
                #pts_len = len(pts)
                immediate_goal.x = goal_x*occ_grid.cell_size + occ_grid.min_pos[0]
                immediate_goal.y = goal_y*occ_grid.cell_size + occ_grid.min_pos[1]
                #print('pub_goal: {}'.format((immediate_goal.x, immediate_goal.y)))
                goal_pub.publish(immediate_goal)
                # print ('help5')       

            if abs(rbt_idx[0] - path[0][0]) < 2 and abs(rbt_idx[1] - path[0][1]) < 2: # If the robot is with 6x6 of goal  
                #print('goal 1: {} reached'.format(goal_idx))
                goal_number += 1
### ARJUN'S EDITS
                if goal_number > len(goals)
                    msg_guider_turtle.stop == True
### ARJUN'S EDITS END
                goal_pos = goals[goal_number]
                
               
            # print(' main.py goal 1: {} reached'.format(goal_idx))
            # print ' main.py: immediate_goal', immediate_goal # rbt_pos is in meters so the goal should also be in meters
            # print ' main.py: goal_number', goal_number
           
            # show the map as a picture
            occ_grid.show_map(rbt_idx, path, goal_idx, pts, immediate_goal)
           
            # increment the time counter
            et = rospy.get_time() - t
            print(et <= 0.4, et)
            t += 0.4
            #print(Project1_aux.myVariable)
            #print(occ_grid.idx2cell((1,2)).is_occupied())
       
       
if __name__ == '__main__':      
    try:
                # parse goals
        goals = sys.argv[1]
        goals = goals.split('|')
        for i in xrange(len(goals)):
            tmp = goals[i].split(',')
            tmp[0] = float(tmp[0])
            tmp[1] = float(tmp[1])
            goals[i] = tmp
        
        # parse cell_size
        cell_size = float(sys.argv[2])
        
        # parse min_pos
        min_pos = sys.argv[3]
        min_pos = min_pos.split(',')
        min_pos = (float(min_pos[0]), float(min_pos[1]))
        
        # parse max_pos
        max_pos = sys.argv[4]
        max_pos = max_pos.split(',')
        max_pos = (float(max_pos[0]), float(max_pos[1]))
        
        main(goals, cell_size, min_pos, max_pos)
        # main(goals)
        # main()
    except rospy.ROSInterruptException:
        pass
