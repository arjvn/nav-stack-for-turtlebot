#!/usr/bin/env python

import roslib, rospy, rospkg
from numpy import *
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, JointState ,Imu
from nav_msgs.msg import Odometry
from std_msgs import *
from std_msgs.msg import Float64MultiArray
from tf.transformations import quaternion_from_euler, euler_from_quaternion, quaternion_multiply
from tf2_msgs.msg import TFMessage
import cv2
import numpy
#import lab2_aux
import sys #lab3_combi

# ================================= CONSTANTS ==========================================        
# let's cache the SIN and POS so we don't keep recalculating it, which is slow
DEG2RAD = [i/180.0*pi for i in xrange(360)] # DEG2RAD[3] means 3 degrees in radians
SIN = [sin(DEG2RAD[i]) for i in xrange(360)] # SIN[32] means sin(32degrees)
COS = [cos(DEG2RAD[i]) for i in xrange(360)]
#MAX_RNG = lab2_aux.MAX_RNG
PATH_PKG = rospkg.RosPack().get_path('pkg') + '/'
PATH_WORLDS = PATH_PKG + 'worlds/'
PI = pi #lab3_combi
INF_RADIUS = 0.2 #lab3_combi
CELL_SIZE = 0.1 #lab3_combi
SQRT2 = sqrt(2) #lab3_combi
TWOPI = 2*PI #lab3_combi
NEAR_RADIUS = 0.2   #lab3_combi
REL_IDX = ((1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (1,-1))
MAX_RNG = 3.5 # search in topic /scan
################ post process ##############################
def post_process(astar_path, occ_grid,los_int, los):
    #print("starting pp")
    astar_path_len = len(astar_path)
    if astar_path_len == 2:
        return [astar_path[0], astar_path[1]], [astar_path[0], astar_path[1]]
    elif astar_path_len == 1:
        return [astar_path[0]], [astar_path[0]]
    #print ("getting seg")
    segs, segs_inf = segment(astar_path , occ_grid)
    #print ("seg_pts")
    segs_pts = turn_points(segs)
    #print ("entering gesegs")
    g_segs_pts = los_points(segs, segs_pts, segs_inf, True, los_int , occ_grid )
    #print ("g seg is " , g_segs_pts)
    #print ("entering s seg")
    s_segs_pts = los_points(segs, segs_pts, segs_inf, False, los_int , occ_grid)
    #print ("entering p_point")
    pts = processed_points(segs, segs_inf, segs_pts, g_segs_pts, s_segs_pts)
    #print ("enter path" )
    path = processed_path(pts, los) # ! 20200227
    path.pop(0)
    return (path , pts)

def segment(astar_path , occ_grid):
    segs = []
    idx = astar_path[0]
    astar_path_len = len(astar_path)
    was_inf = occ_grid.idx2cell(idx[0],idx[1]).is_inflation()
    #was_inf = is_inflation(idx)
    seg = [idx]
    segs_inf = [was_inf]
    for a in xrange(1, astar_path_len):
        idx = astar_path[a]
        #is_inf = is_inflation(idx)
        is_inf = occ_grid.idx2cell(idx[0],idx[1]).is_inflation()
        if was_inf != is_inf:
            #print ("inflation found ?")
            segs.append(seg)
            seg = [idx]
            segs_inf.append(is_inf)
        else:
            seg.append(idx)
        was_inf = is_inf

    segs.append(seg)
    return segs, segs_inf




def turn_points(segs):
    segs_pts = []; seg_pts = []

    for seg in segs:
        seg_pts = [0]
        seg_len = len(seg)
        segs_pts_append = segs_pts.append
        seg_pts_append = seg_pts.append
        if seg_len == 1:
            segs_pts_append(seg_pts)
            continue
        elif seg_len > 2:
            i0, j0 = seg[1]
            i1, j1 = seg[0]
            di0 = i0 - i1; dj0 = j0 - j1
            for b in xrange(2, seg_len):
                i1, j1 = seg[b]
                di1 = i1 - i0; dj1 = j1 - j0
                if di0 != di1 or dj0 != dj1:
                    seg_pts_append(b - 1)
                di0 = di1; dj0 = dj1
                i0 = i1; j0 = j1
        seg_pts_append(seg_len - 1)
        segs_pts_append(seg_pts)
    return segs_pts

def los_points(segs, segs_pts, segs_inf, forward, los_int , occ_grid):
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
        at_end = lambda p, length : p == -1 # does not depend on length 
        aft_end = lambda p, length : p < -1 # does not depend on length
    # initialise new list to hold all LOS pts in all segments
    new_segs_pts = []
    # for every segment
    segs_len = len(segs)
    for a in xrange(segs_len):
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
            #print("lp5")        
            new_segs_pts.append(new_seg_pts)
        else: # seg_pts_len > 2
            seg = segs[a]
            p = get_first(seg_pts_len) # first turning pt, actually just the first in seg
            start_idx = seg[seg_pts[p]] # get the idx of the first turning pt
            p = get_third(seg_pts_len) # 2 for forward, seg_pts_len - 3 for backward
            q = seg_pts[p]
            end_idx = seg[q] # get the idx of the third turning pt.
            # set LOS
            direct_LOS = True
            los_indices = los_int.calculate(start_idx, end_idx)
            #print ("los_indices is " , los_indices)
            l = 0
            while True:
                l += 1

                len_los_indices = len(los_indices)
                if len_los_indices == 1:
                    l -= 1

                #print ("len_los is and l is  " , len_los_indices , l )
                #print ("lost indices are" ,  los_indices)
                idx = los_indices[l]

                #print 
                if idx == end_idx:
                    new_seg_pts.append(q)
                    p = next_p(p, direct_LOS) # get the next turning point in each segment
                    if at_end(p, seg_pts_len): # == seg_pts_len for forward, == -1 for backward
                        p = get_last(seg_pts_len)
                        new_seg_pts.append(seg_pts[p])
                        break
                    elif aft_end(p, seg_pts_len): # > seg_pts_len for forward, < -1 for backward
                        break
                    # Else
                    else: # some turn points left in segment
                        direct_LOS = True
                        start_idx = end_idx
                        q = seg_pts[p] # get the next turn pt.
                        end_idx = seg[q] # get the next turn pt. index
                        #print(end_idx)
                        los_indices = los_int.calculate(start_idx, end_idx)
                        l = 0
                        continue
                # check if los crosses an inflated cell
                if occ_grid.idx2cell(idx[0],idx[1]).is_inflation():
                    direct_LOS = False
                    # set the end idx one cell backward in seg
                    q = next_q(q) # q-1 for forward, q+1 for backward
                    end_idx = seg[q]
                    los_indices = los_int.calculate(start_idx, end_idx)
                    l = 0
       
            # same indent level as comment: # skip if segment is inflated
            new_segs_pts.append(new_seg_pts)
    return new_segs_pts



def processed_points(segs, segs_inf, segs_pts, g_segs_pts, s_segs_pts):
    pts = []
    # for every segment
    segs_len = len(segs)
    for a in xrange(segs_len):
        seg = segs[a]

        seg_pts = segs_pts[a] # ! 20200227
        # preserve turning points on inflated segments
        if segs_inf[a]:
            for q in seg_pts: # ! 20200227
                pts.append(seg[q])
            continue
        g_seg_pts = g_segs_pts[a] # ! 20200227
        s_seg_pts = s_segs_pts[a] # ! 20200227
        seg_pts_len = len(g_seg_pts)
        p = 0; q = seg_pts_len - 1 # s_seg_pts and g_seg_pts length are the same
        #print ("q  and len(s_seg_pts ) is  " , q , seg_pts_len , len(s_seg_pts))
        while p < seg_pts_len:
            g = g_seg_pts[p]
            s = s_seg_pts[q]
            if g == s:

                pts.append((float(seg[g][0]) , float(seg[g][1]))) # convert the int index to float index
                p += 1 # ! 20200227
                q -= 1 # ! 20200227
                continue # ! 20200227
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
                idx  = ((b1*c0 - b0*c1)/d, (a0*c1 - a1*c0)/d) # float form
                pts.append(idx)
            p += 1
            q -= 1
    return pts    
    
def processed_path(pts, los):
    start_idx = pts[0] # pts should be in float form
    path = [start_idx] # assuming los does not return the start_idx
    # for every segment
    pts_len = len(pts)
    for p in xrange(1, pts_len):
            end_idx = pts[p]
            for idx in los.calculate(start_idx, end_idx):
                # assuming los does not return the start_idx
                path.append(idx)
            start_idx = end_idx
    return path



    #################### post process end ######################################
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
    def reset_for_planner(self, goal_idx_i, goal_idx_j):
        """ Resets the cells for every run of non-dynamic path planners
        Parameters:
            goal_idx (tuple of int64):  Index of goal cell
        """
        self.g_cost = Distance(inf, inf)
        self.h_cost = Distance.from_separation(self.idx, (goal_idx_i,goal_idx_j))
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
        #round it and give it fixed numbers so won't eat so much processing time
        #lo = log(0.3/0.7) = -0.847297860387
        #locc = log(0.8/0.2) = 1.38629436112
        #lfree = -locc
        zhng the value abit and arrived at 2.4 and 0.5 added a hard limiter
        of 40 and -20 to because occupied grows 5 times faster than free
        """
#        lab2_aux.set_occupancy(self, occupied)

        if occupied:
            if self.occ < 30: # best is 2.4 so far
                self.occ += 2.5 # locc-lo = 2.23359222151
        else:
            if self.occ > -20: # best is 0.5 so far
                self.occ -=  0.5 #lfree-lo  = -0.538996500733
        
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
        return self.occ > 10
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
        return self.occ < -10 
    def is_unknown(self):
        """ Returns True if the cell's occupancy is unknown
        Returns:
            bool : True if cell's occupancy is unknown, False otherwise
        """
        return self.occ >= -10 and self.occ <= 10
    def is_planner_free(self):
        """ Returns True if the cell is traversable by path planners
        Returns:
            bool : True if cell is unknown, free and not inflated, 
        """
        return self.occ <= 10 and not self.inf
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
        return 'Cell{} occ:{}, f:{:6.2f}, g:{:6.2f}, h:{:6.2f}, visited:{}, parent:{}'  .format(self.idx, self.occ, self.f_cost.total, self.g_cost.total, self.h_cost.total, self.visited, \
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
        di = int(round((max_pos[0] - min_pos[0])/cell_size))
        dj = int(round((max_pos[1] - min_pos[1])/cell_size))
        self.cell_size = cell_size
        self.min_pos = min_pos
        self.max_pos = max_pos
        di += 1; dj += 1
        self.num_idx = (di, dj) # number of (rows, cols)
        self.i2x = lambda idx2pos_x: float (idx2pos_x * cell_size + min_pos[0]) #index to pos (x variable)
        self.j2y = lambda idx2pos_y: float (idx2pos_y * cell_size + min_pos[1]) #index to pos (y variable)
        self.x2iE = lambda pos2idxxE: (pos2idxxE - min_pos[0])/cell_size #pos to idx w/o rounded (Exact) (x variable)
        self.y2jE = lambda pos2idxyE: (pos2idxyE - min_pos[1])/cell_size #pos to idx w/o rounded (Exact) (y variable)
        self.x2i = lambda pos2idxx: int(round((pos2idxx - min_pos[0])/cell_size)) #pos to idx rounded (x variable)
        self.y2j = lambda pos2idxy: int(round((pos2idxy - min_pos[1])/cell_size)) #pos to idx rounded (y variable)
        self.cells = [[Cell((i,j), initial_value) for j in xrange(dj)] for i in xrange(di)]
        self.mask_inflation = gen_mask(cell_size, inflation_radius)
        # CV2 inits
        self.img_mat = full((di, dj, 3), uint8(127))
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('img', di*2, dj*2) # so each cell is 2px*2px
        '''
        #For checking against map update , you may want to use a list of Boolean lists
        #hence generate a boolean list for of di by  dj
        # https://stackoverflow.com/questions/18123965/why-is-if-true-slower-than-if-1
        # however use 1 0 instead of true false according to this link as is faster
        path_old to store the old path and need_path== 1 to mean need new path
        need_path = 0 means dun nid need path. boolean_list is a list of 1 and 0 
        with the size di by dj to cater to the whole map
        '''
        self.boolean_list = [ [ 0 for idx_j in xrange(dj) ] for idx_i in xrange(di) ] 
        self.need_path = 1
        self.path_old = []

    '''
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
            return (int(round(idx[0])), int(round(idx[1])))
        return idx
    '''
    def idx_in_map(self, i,j): # idx must be integer
        """ Checks if the given index is within map boundaries
        Parameters:
            idx (tuple of int64): Index tuple (i, j) to be checked
        Returns:
            bool: True if in map, False if outside map
        """
        return i >= 0 and i < self.num_idx[0] and j >= 0 and j < self.num_idx[1]
    def idx2cell(self, idx_i, idx_j, is_int=False):
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
            idx = (int(round(idx_i)), int(round(idx_j)))
        if self.idx_in_map(idx_i , idx_j):
            return self.cells[idx_i][idx_j]
        return None
        
    def update_at_idx(self, idx_i , idx_j, occupied):
        """ Updates the cell at the index with the observed occupancy
        Parameters:
            idx (tuple of int64): The Index of the cell to update
            occupied (bool):    If True, the cell is currently observed to be occupied. 
                                False if free.
        if occupancy changed that means occupied has been found ?
        hence check this occupied cell has a path on it if 1, need_path = 1
        next do the same with the inflation
        use conditional loop to check the variable need_path== 1
        then update self.need_path, if not it will stop replanning due to update_at_idx 
        being run 1st then the planner section in the main loop
        """
        ok = self.idx_in_map
        #need_path = self.need_path
        boolean_list = self.boolean_list
        #print ("boolean_list " ,boolean_list )
        need_path = 0
        # return if not in map
        if not ok(idx_i , idx_j):
            return
        c = self.cells
        cell = c[idx_i][idx_j]
        
        # update occupancy
        was_occupied = cell.is_occupied()
        cell.set_occupancy(occupied)
        
        # check if the cell occupancy state is different, and update the masks accordingly 
        # (much faster than just blindly updating regardless of previous state)
        is_occupied = cell.is_occupied()
        #if is_occupied:
            #if boolean_list[idx[0]][idx[1]]:
                #print ("truth1")
                #need_path = 1
        mask_inflation = self.mask_inflation
        if was_occupied != is_occupied:
            if boolean_list[idx_i][idx_j] == 1:
                #print ("truth 1")
                need_path = 1
            # if path lies over here we replan
            for rel_idx in mask_inflation:
                i = rel_idx[0] + idx_i
                j = rel_idx[1] + idx_j
                #mask_idx = (i,j)
                if ok(i,j): # cell in map
                    c[i][j].set_inflation((idx_i, idx_j) , is_occupied)
                    # if inflation lies on path we replan
                    if boolean_list[i][j] == 1:
                        #print ("truth 2")
                        need_path = 1
        if need_path ==1:                    
            self.need_path = 1                    
    def update_at_pos(self, pos, occupied):
        """ Updates the cell at the position with the observed occupancy
        Parameters:
            pos (tuple of float64): The position of the cell to update
            occupied (bool):    If True, the cell is currently observed to be occupied. 
                                False if free.
        """
        self.update_at_idx(self.x2i(pos[0]),self.y2j(pos[1]), occupied)
    def update_path(self, path):
        '''
        set the old path as 0 to signify that particular cell's index does not have a path on it
        update the boolean_list with the new path fed in from path to signify this cells are where 
        the path is on
        use it to cross reference at update_at_idx
        '''
        ### set boolean list as 0 for the old path so it won't call for request
        for idx in self.path_old:
            self.boolean_list[idx[0]][idx[1]] = 0
        ## set boolean list as 1 for new path fed in from post process
        for idx in path:
            self.boolean_list[idx[0]][idx[1]] = 1
        self.path_old = path
        
    def show_map(self, rbt_idx_i , rbt_idx_j , path=None, goal_idx_i = None , goal_idx_j  = None):
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
                    #img_mat[i, j, :] = (255, 255, 255) # white
                      img_mat[i,j,0] = 255
                      img_mat[i,j,1] = 255
                      img_mat[i,j,2] = 255
                      
                elif cell.is_inflation():
                    #img_mat[i, j, :] = (180, 180, 180) # light gray
                      img_mat[i,j,0] = 180
                      img_mat[i,j,1] = 180
                      img_mat[i,j,2] = 180
                    
                elif cell.is_free():
                    #img_mat[i, j, :] = (0, 0, 0) # black
                      img_mat[i,j,0] = 0
                      img_mat[i,j,1] = 0
                      img_mat[i,j,2] = 0
                    
        if path is not None:
            path_len = len(path)
            for k in xrange(path_len):
#                idx = path[k]; next_idx = path[k+1]
                ta = path[k]; i=ta[0] ; j = ta[1]
                #img_mat[i, j, :] = (0, 0, 255) # red
                img_mat[i,j,0] = 0
                img_mat[i,j,1] = 0
                img_mat[i,j,2] = 255                
#                cv2.line(img_mat, idx, next_idx, (0,0,255), 1)
            
            
        # color the robot position as a crosshair
        #img_mat[rbt_idx[0], rbt_idx[1], ;] = (0, 255, 0) # green
        img_mat[rbt_idx_i , rbt_idx_j, 0] = 0 # green
        img_mat[rbt_idx_i, rbt_idx_j, 1] = 255 # green
        img_mat[rbt_idx_i, rbt_idx_j, 2] = 0 # green
        
        if goal_idx_i is not None:
            #img_mat[goal_idx[0], goal_idx[1], :] = (255, 0, 0) # blue
            img_mat[goal_idx_i, goal_idx_j, 0] = 255 # blue
            img_mat[goal_idx_i, goal_idx_j, 1] = 0 # blue
            img_mat[goal_idx_i, goal_idx_j, 2] = 0 # blue            

        # print to a window 'img'
        cv2.imshow('img', img_mat)
        cv2.waitKey(10)

# =============================== PLANNER CLASSES =========================================   
class Distance:
    def __init__(self, ordinals=0, cardinals=0):
        """ The constructor for more robust octile distance calculation. 
            Stores the exact number of ordinals and cardinals and calculates
            the approximate float value of the resultant cost in .total
        Parameters:
            ordinals (int64): The integer number of ordinals in the distance
            cardinals (int64): The integer number of cardinals in the distance
        """
        self.axes = [float(ordinals), float(cardinals)]
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
        self.open_list = OpenList()

    def get_path(self, start_idx_i,start_idx_j, goal_idx_i,goal_idx_j):
        """ Returns a list of indices that represents the octile-optimal
            path between the starting index and the goal index
        Parameters:
            start_idx (tuple of int64): the starting index
            goal_idx (tuple of int64): the goal index
        Returns:
            list of tuple of int64: contains the indices in the optimal path
        """
        occ_grid = self.occ_grid
        open_list = self.open_list
        # get number of rows ni (x) and number of columns nj (y)

        ni, nj = occ_grid.num_idx
        path = []
        # resets h-cost, g-cost, update and occ for all cells

        for i in xrange(ni):
            for j in xrange(nj):
                occ_grid.idx2cell(i, j).reset_for_planner(goal_idx_i,goal_idx_j)
                # ! use occ_grid.idx2cell() and the cell's reset_for_planner()

        # put start cell into open list
        start_cell = occ_grid.idx2cell(start_idx_i, start_idx_j)

        start_cell.set_g_cost(Distance(0, 0))
        open_list.add(start_cell)
        # ! get the start cell from start_idx
        # ! set the start cell distance using set_g_cost and Distance(0, 0)
        # ! add the cell to open_list
        #print ("astar stuck here 1")
        # now we non-recursively search the map
        while open_list.not_empty():
            cell = open_list.remove()
            # skip if already visited, bcos a cheaper path was already found
            if cell.visited:
                continue
            cell.visited = True
            # ! set the cell as visited
            
            # goal
            if cell.idx == (goal_idx_i,goal_idx_j):
                while True:
                    path.append(cell.idx)
                    cell = cell.parent
                    if cell is None:
                        break 
                    # ! append the cell.idx onto path
                    # ! let cell = cell's parent
                    # ! if cell is None, break out of the while loop
                break # breaks out of the loop: while open_list.not_empty()
            #print ("astar stuck here 2")
    
            # if not goal or not visited, we try to add free neighbour cells into the open list    
            for nb_cell in self.get_free_neighbors(cell):
                tentative_g_cost = cell.g_cost + Distance.from_separation(cell.idx, nb_cell.idx)
                if tentative_g_cost < nb_cell.g_cost:
                    nb_cell.set_g_cost(tentative_g_cost)
                    nb_cell.parent = cell
                    open_list.add(nb_cell)
                    #print ("astar stuck here 3")

                # ! calculate the tentative g cost of getting from current cell (cell) to neighbouring cell (nb_cell)...
                # !     use cell.g_cost and Distance.from_separation()
                # ! if the tentative g cost is less than the nb_cell.g_cost, ...
                # !     1. assign the tentative g cost to nb_cell's g cost using set_g_cost
                # !     2. set the nb_cell parent as cell
                # !     3. add the nb_cell to the open list using open_list.add()
        #print ("astar stuck here 4")

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
            nb_idx_i = rel_idx[0] + idx[0]
            nb_idx_j = rel_idx[1] + idx[1]            
            nb_cell = occ_grid.idx2cell(nb_idx_i, nb_idx_j)
            if nb_cell is not None and nb_cell.is_planner_free():
                neighbors.append(nb_cell)
        #print ("neighbour stuck")

        return neighbors


class OpenList():

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
        i = 0
        nl = len(l)
        while i < nl:
            # ! get the cell (list_cell) in the index (i) of the open list (l)
            # ! now if the cell's f_cost is less than the list_cell's f_cost, ...
            # !     or if the cell's f_cost = list_cell's f_cost but ...
            # !     cell's h_cost is less than the list_cell's h_cost...
            # !     we break the loop (while i < nl)
            list_cell = l[i]
            if cell.f_cost < list_cell.f_cost or cell.f_cost == list_cell.f_cost and cell.h_cost < list_cell.h_cost:
                break
            # increment the index                
            i += 1
        # insert the cell into position i of l
        #print ("add stuck")
        l.insert(i, cell)

    def remove(self):
        """ Removes and return the cheapest cost cell in the open list
        Returns:
            Cell: the cell with the cheapest f-cost followed by h-cost
        """
        # return the first element in self.l
        
        return self.l.pop(0)

    def not_empty(self):
        return not not self.l # self.l is False if len(self.l) is zero ==> faster

    def __str__(self):
        l = self.l
        s = ''
        for cell in l:
            s += ('({:3d},{:3d}), F:{:6.2f}, G:{:6.2f}, H:{:6.2f}\n').format(cell.idx[0], cell.idx[1], cell.f_cost.total, cell.g_cost.total, cell.h_cost.total)

        return s
    
# =============================== SUBSCRIBERS =========================================  
def subscribe_move(msg):
    global msg_move
    t = msg.data
    msg_move[0] = t[0] # rx
    msg_move[1] = t[1] # ry
    msg_move[2] = t[2] # ro
    msg_move[3] = t[3] # positional error

############ should remove ? #############################
'''
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
'''    
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

def subscribe_wheels(msg):
    global rbt_wheels
    rbt_wheels = (msg.position[1], msg.position[0])    
# ================================== OdometryMM ========================================    
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
        vt = ((self.WR/ (2 * dt)) * (dwr + dwl)) * 0.3 + (self.v + rbt_imu_a * dt) * 0.7
        wt = ((self.WR/ (self.L *dt)) * (dwr - dwl)) * 0.35 + rbt_imu_w * 0.65
        dphi = self.o + ((self.WR /self.L)* (dwr - dwl))
        dphi = (dphi * 0.1 + rbt_imu_o * 0.9)
        if abs(wt) < 0.05:
            self.x =self.x +  vt * dt * cos(self.o)
            self.y = self.y +  vt * dt * sin(self.o)
        else:
            rt = vt / wt
            self.x += -rt * sin(self.o) + rt * sin(dphi)
            self.y += rt * cos(self.o) - rt * cos(dphi)
            self.o = dphi #move this out of loop so we can make use of imu for both cases
        self.t = self.t + dt
        self.wl = wheels[0]
        self.wr = wheels[1]
        self.v = vt
        return (self.x, self.y, self.o)
# ================================== LOS ========================================
class LOS:
    def __init__(self, map):
        self.x2i = map.x2i
        self.y2j = map.y2j
#        self.pos2idx = map.pos2idx # based on the map (occ_grid) it return's the map index representation of the position pos
        # use self.pos2idx(pos, False) to return the exact index representation, including values that are less than 1.
        # use self.pos2idx(pos) to return the integer index representation, which is the rounded version of self.pos2idx(pos, False)

# general line algorithm

    def calculate(self, start_pos_i, start_pos_j, end_pos_i, end_pos_j):
        # sets up the LOS object to prepare return a list of indices on the map starting from start_pos (world coordinates) to end_pos (world)
        # start_pos is the robot position.
        # end_pos is the maximum range of the LIDAR, or an obstacle.
        # every index returned in the indices will be the index of a FREE cell
        # you can return the indices, or update the cells in here
#        start_idx = (self.x2i(start_pos[0]), self.y2j(start_pos[1])) # <-- use start_idx. Don't use start_pos. 
        start_idx_i = float (self.x2i(start_pos_i)); start_idx_j =  float (self.y2j(start_pos_j))
#        end_idx = (self.pos2idx(end_pos)) # <-- use end_idx. Don't use end_pos.
        end_idx_i = float (self.x2i(end_pos_i)) ; end_idx_j = float (self.y2j(end_pos_j))
        indices = [] # init an empty list
        indices.append((int(start_idx_i),int(start_idx_j))) # append the starting index into the cell
        # get diff
        di = end_idx_i-start_idx_i # i is along x-axis
        dj =  end_idx_j-start_idx_j # j is along y-axis
        

        # assign short and long
        if abs(di)>abs(dj):
            dl = di
            ds = dj
            get_idx = lambda(l,s) : (l,s)
        else:
            dl = dj
            ds = di
            get_idx = lambda(l,s) : (s,l)
        (italic_l, italic_s) = get_idx((start_idx_i,start_idx_j))
        (italic_lf,italic_sf) = get_idx((end_idx_i,end_idx_j))
        # get integer (index) representation for accessing map
        (l,s) = (round(italic_l),round(italic_s)) 
        (lf,sf) = (round(italic_lf),round(italic_sf))
        
        # get signs and increments
        delta_s = sign(ds)
        delta_l = sign(dl)        
        if dl == 0: # added from forum's suggestion
            grad_s = 0
            err_s = 0
            lamda = 0
        else:
            grad_s= ds / abs(dl)
            err_s = italic_s - s
            lamda =abs(grad_s)*(0.5+(italic_l-l)*delta_l)-0.5

        #grad_s = ds/abs(dl)
        # get error
        #err_s = italic_s - s
        # get lambda
        #lamda = abs(ds/dl)*(0.5+(italic_l-l)*delta_l)-0.5
        # get error checker
        if ds >= 0:
            has_big_err = lambda err : err >= 0.5
        else:
            has_big_err = lambda err : err < -0.5
        # propagate 
        while (l,s) != (lf,sf):
            l += delta_l
            err_s += grad_s
            if has_big_err(err_s):
                err_s -= delta_s
                s += delta_s
                # previous cell(s)
                italic_lamda = err_s*delta_s
                if italic_lamda < lamda: #short direction
                    indices.append(get_idx((int(l),int(s-delta_s)))) #get int value. round will still remain as float: xx.0
                    if (l,s-delta_s) == (lf,sf): break 
                elif italic_lamda > lamda: #long direction
                    indices.append(get_idx((int(l-delta_l),int(s))))
                    if(l-delta_l,s) == (lf,sf): break
                else: #both direction
                    indices.append(get_idx((int(l-delta_l),int(s))))
                    if (l-delta_l,s) == (lf,sf): break
                    indices.append(get_idx((int(l),int(s-delta_s))))
                    if (l,s-delta_s) == (lf,sf): break
            # current cell        
            indices.append(get_idx((int(l),int(s))))
        return indices
######################################### general LOS#################

class GeneralLOS:
    def __init__(self):
        self.i = 0

# general line algorithm

    def calculate(self, start_idx, end_idx):
        # sets up the LOS object to prepare return a list of indices on the map starting from start_pos (world coordinates) to end_pos (world)
        # start_pos is the robot position.
        # end_pos is the maximum range of the LIDAR, or an obstacle.
        # every index returned in the indices will be the index of a FREE cell
        # you can return the indices, or update the cells in here
        #start_idx = float64(self.pos2idx(start_pos)) # <-- use start_idx. Don't use start_pos. 
        #end_idx = float64(self.pos2idx(end_pos)) # <-- use end_idx. Don't use end_pos.
        indices = [] # init an empty list
        indices.append((int(start_idx[0]), int(start_idx[1]))) # append the starting index into the cell
        # get diff
        di = end_idx[0]-start_idx[0] # i is along x-axis
        dj =  end_idx[1]-start_idx[1] # j is along y-axis
        
#        start_idx_i = int(start_idx[0])
#        start_idx_j = int(start_idx[1])
#        end_idx_i = int(end_idx[0])
#        end_idx_j = int(end_idx[1])        #indices.append(int64start_idx)) # append the starting index into the cell
#        indices.append((start_idx_i , start_idx_j)) # append the starting index into the cell
        #print("incidces")
        #print(indices) 
        # get diff
#        di = end_idx_i-start_idx_i # i is along x-axis
#        dj =  end_idx_j-start_idx_j # j is along y-axis
        
        # assign short and long
        if abs(di)>abs(dj):
            dl = di
            ds = dj
            get_idx = lambda(l,s) : (l,s)
        else:
            dl = dj
            ds = di
            get_idx = lambda(l,s) : (s,l)
        (italic_l, italic_s) = get_idx((start_idx[0],start_idx[1]))
        (italic_lf,italic_sf) = get_idx((end_idx[0],end_idx[1]))
        # get integer (index) representation for accessing map
        (l,s) = (round(italic_l),round(italic_s)) 
        (lf,sf) = (round(italic_lf),round(italic_sf))

        
        # get signs and increments
        delta_s = sign(ds)
        delta_l = sign(dl)        
        if dl == 0: # added from forum's suggestion
            grad_s = 0
            err_s = 0   
            lamda = 0
        else:
            grad_s= ds / abs(dl)
            err_s = italic_s - s
            lamda =abs(grad_s)*(0.5+(italic_l-l)*delta_l)-0.5

        #grad_s = ds/abs(dl)
        # get error
        #err_s = italic_s - s
        # get lambda
        #lamda = abs(ds/dl)*(0.5+(italic_l-l)*delta_l)-0.5
        # get error checker
        if ds >= 0:
            has_big_err = lambda err : err >= 0.5
        else:
            has_big_err = lambda err : err < -0.5
        # propagate 
        while (l,s) != (lf,sf):
            l += delta_l
            err_s += grad_s
            if has_big_err(err_s):
                err_s -= delta_s
                s += delta_s
                # previous cell(s)
                italic_lamda = err_s*delta_s
                if italic_lamda < lamda: #short direction
                    indices.append(get_idx((int(l),int(s-delta_s)))) #get int value. round will still remain as float: xx.0
                    if (l,s-delta_s) == (lf,sf): break 
                elif italic_lamda > lamda: #long direction
                    indices.append(get_idx((int(l-delta_l),int(s))))
                    if(l-delta_l,s) == (lf,sf): break
            # current cell        
            indices.append(get_idx((int(l),int(s))))
        return indices
        
   ##################general int los###############################

class GeneralIntLOS:

# general line algorithm for int
    def __init__(self):
        self.i = 0
        # use self.pos2idx(pos, False) to return the exact index representation, including values that are less than 1.
        # use self.pos2idx(pos) to return the integer index representation, which is the rounded version of self.pos2idx(pos, False)
    def calculate(self, start_idx, end_idx):
        # sets up the LOS object to prepare return a list of indices on the map starting from start_pos (world coordinates) to end_pos (world)
        # start_pos is the robot position.
        # end_pos is the maximum range of the LIDAR, or an obstacle.
        # every index returned in the indices will be the index of a FREE cell
        # you can return the indices, or update the cells in here
        #start_idx = int64(self.pos2idx(start_pos)) # <-- use start_idx. Don't use start_pos. 
        #end_idx = int64(self.pos2idx(end_pos)) # <-- use end_idx. Don't use end_pos.
        indices = [] # init an empty list
        start_i = int(start_idx[0])
        start_j = int(start_idx[1])
        end_idx_i = int(end_idx[0])
        end_idx_j = int(end_idx[1])        #indices.append(int64start_idx)) # append the starting index into the cell
        indices.append((start_i , start_j)) # append the starting index into the cell
        #print("incidces")
        #print(indices) 
        # get diff
        di = end_idx_i-start_i # i is along x-axis
        dj =  end_idx_j-start_j # j is along y-axis
        #print("end idx")
        #print(end_idx)
        #print("start idx")
        #print(start_idx)
        # assign short and long
        if abs(di)>abs(dj):
            dl = di
            ds = dj
            get_idx = lambda(l,s) : (l,s)
        else:
            dl = dj
            ds = di
            get_idx = lambda(l,s) : (s,l)
            
        (l, s) = get_idx((start_i,start_j))
        (lf,sf) = get_idx((end_idx_i,end_idx_j))
        
        #print("i")
        #print(l)
        #print("s")
        #print(s)
        
        #print("lf")
        #print(lf)
        
        #print("dl")
        #print(dl)
        #print("ds")
        #print(ds)
     
        # get signs and increments
        abs_dl = abs(dl)
        delta_s = sign(ds)
        delta_l = sign(dl)
        grad_s = 2*ds
        eta_sl = 2 * abs_dl * delta_s
        #print("delta_s")
        #print(delta_s)
        #print("delta_l")
        #print(delta_l)
        # get error
        err_sl = 0
        # get lambda
        lamda = abs(ds) - abs_dl
        # get error checker
        if ds >= 0:
            has_big_err = lambda err : err >= abs_dl
        else:
            has_big_err = lambda err : err < -abs_dl
        # propagate
        #print("l")
        #print(l-1)     
        #print("im here")
        #print(lf)        
        while (l,s) != (lf,sf):
            #print("sf")
            #print(s)  
            l += delta_l
            err_sl+= grad_s
            if has_big_err(err_sl):
                err_sl -= eta_sl
                s += delta_s
                # previous cell(s)
                italic_lamda = err_sl*delta_s
                #print(lamda)
                if italic_lamda < lamda: #short direction
                    indices.append(get_idx((int(l),int(s-delta_s)))) #get int value. round will still remain as float: xx.0
                    if (l,s-delta_s) == (lf,sf): break 
                elif italic_lamda > lamda:  #long direction
                    indices.append(get_idx((int(l-delta_l),int(s))))
                    if(l-delta_l,s) == (lf,sf): break
            # current cell        
            indices.append(get_idx((int(l),int(s))))
        '''        
        print("indices")
        print(indices)
        '''
        return indices
# ================================== JPS ========================================
# ================================== PUBLISHERS ========================================
# Define the LIDAR maximum range


def inverse_sensor_model(rng, deg, pose):
    # degree is the bearing in degrees # convert to radians #deg is phi_k
    # range is the current range data at degree
    # pose is the robot 3DOF pose, in tuple form, (x, y, o) # o is phi_t
    ta1 = pose; x=ta1[0] ; y = ta1[1] ; o  = ta1[2]
    xk = x + rng * cos(o + DEG2RAD[deg]) 
    yk = y + rng * sin(o + DEG2RAD[deg]) 
    return (xk, yk)
# =================================== OTHER METHODS =====================================
def gen_mask(cell_size, radius):
    """ Generates the list of relative indices of neighboring cells which lie within 
        the specified radius from a center cell
    Parameters:
        radius (float64): the radius 
    """
    #return lab2_aux.gen_mask(cell_size, radius)
    
    nb_list = []
    num_cell_idx = int(radius/cell_size)/2#number of cells within radius in idx, not meter form
    for i in xrange(-num_cell_idx, num_cell_idx+1): #loop through (-a,a+1) horizontaly and vertically
        for j in xrange (-num_cell_idx, num_cell_idx+1):
            #nb_cell = (i,j)
            nb_list.append((i,j))
    
    return nb_list

# ================================ BEGIN ===========================================
#MotionModel = lab2_aux.OdometryMM
MotionModel = OdometryMM
#LOS = lab2_aux.GeneralLOS
#inverse_sensor_model = lab2_aux.inverse_sensor_model
#PathPlanner = lab2_aux.Astar
PathPlanner = Astar
#PathPlanner = JPS
def main(goals, cell_size, min_pos, max_pos , start_pose): #lab3_combi
    # ---------------------------------- INITS ----------------------------------------------
    CELL_SIZE = cell_size
    # init node
    rospy.init_node('main')
 
    # Set the labels below to refer to the global namespace (i.e., global variables)
    # global is required for writing to global variables. For reading, it is not necessary
    global rbt_imu_w , rbt_scan, read_scan, write_scan, msg_move , rbt_wheels #rbt_true , rbt_control   #lab3_combi
    
    # Initialise global vars with NaN values 
    # nan and inf are imported from numpy. If you use "import numpy as np", then nan is np.nan, and inf is np.inf.
    rbt_scan = [nan]*360 # a list of 360 nans
    #rbt_true = [nan]*3
    rbt_wheels = [nan]*2
    rbt_scan = None

    read_scan = False
    write_scan = False
    msg_move = [-1. for i in xrange(4)]
    rbt_imu_w = None
    
    # Subscribers
    rospy.Subscriber('scan', LaserScan, subscribe_scan, queue_size=1) 
#    rospy.Subscriber('tf', TFMessage, subscribe_true, queue_size=1)
    rospy.Subscriber('joint_states', JointState, subscribe_wheels, queue_size=1)

    #rospy.Subscriber('tf', TFMessage, subscribe_true, queue_size=1) #lab3_combi
    #rospy.Subscriber('joint_states', JointState, subscribe_wheels, queue_size=1) #lab3_combi
    rospy.Subscriber('move', Float64MultiArray, subscribe_move, queue_size=1)
    rospy.Subscriber('imu', Imu, subscribe_imu, queue_size=1)
    
    
      # Publishers
    # ~ publish to topic 'lab3', a float64multiarray message
    publisher_main = rospy.Publisher('main', Float64MultiArray, latch=True, queue_size=1)
    msg_main = [0. for i in xrange(3)] 
    # ~ [0] operating mode: 0. is run, 1. is stop running and exit.
    # ~ [1] px: the x position of the target for the robot to pursue
    # ~ [2] py: the y position of the target for the robot to pursue
    msg_m = Float64MultiArray()
    # cache the part where we want to modify for slightly faster access
    msg_m.data = msg_main
    # publish first data for main node to register
    publisher_main.publish(msg_m)
    
    # Wait for Subscribers to receive data.
    #while isnan(rbt_scan[0]) or isnan(rbt_true[0]) or isnan(rbt_wheels[0]):
      #  pass
    
    while (rbt_scan is None or msg_move[0] == -1. or isnan(rbt_wheels[0])  or rbt_imu_w is None ) and not rospy.is_shutdown():
        pass  

   
    if rospy.is_shutdown():
        return
        
        
    # Data structures
    
    #occ_grid = OccupancyGrid((-2,-3), (5,4), 0.1, 0, 0.2)
    occ_grid = OccupancyGrid(min_pos, max_pos, CELL_SIZE, 0, INF_RADIUS)
    los = LOS(occ_grid)
    los_calc = los.calculate
#    motion_model = OdometryMM(start_pose, rbt_wheels, 0.16, 0.066)  #lab3_combi
    planner = PathPlanner(occ_grid) 
    planner_get_path = planner.get_path
    #goal_pos = (1.5,1.5)  #lab3_combi  #lab3_combi
    
    #los_int = lab3_aux.GeneralIntLOS()
    los_int = GeneralIntLOS()
    los_pp = GeneralLOS()
    update_at_pos = occ_grid.update_at_pos
    update_at_idx = occ_grid.update_at_idx
#    pos2idx =occ_grid.pos2idx
    x2i = occ_grid.x2i
    y2j = occ_grid.y2j
    x2iE = occ_grid.x2iE
    y2jE = occ_grid.y2jE
    #idx2pos = occ_grid.idx2pos
    i2x = occ_grid.i2x
    j2y = occ_grid.j2y
    post_processes = post_process
    show_map = occ_grid.show_map
    inverse_sensor_model_cache = inverse_sensor_model
    #pos2idx_float = occ_grid.pos2idx_float
    # get the first goal pos
    gx = goals[0][0]; gy = goals[0][1]
    # number of goals (i.e. areas)
    g_len = len(goals)
    # get the first goal idx
    #gi = occ_grid.x2i(gx); gj = occ_grid.y2j(gy)
#    goal_pos = (gx,gy) 
    #goal_pos = pos2idx((gx,gy))
    #goal_idx = pos2idx(goal_pos)
    goal_idx_i = x2i(gx);goal_idx_j = y2j(gy)
    # set the goal number as zero
    g = 0
    px = 0 #initialise px and py and path
    py = 0
    path = []
    # ---------------------------------- BEGIN ----------------------------------------------
    t = rospy.get_time()
    update = 0
    while (not rospy.is_shutdown()): # required to Keyboard interrupt nicely
        
        if (rospy.get_time() > t): # every 50 ms
            # get position #lab3_combi
            rx = msg_move[1]
            ry = msg_move[2]
            ro = msg_move[3]
            # get scan
            scan = get_scan()
            
            #motion_model.calculate(rbt_wheels, rbt_imu_w, rbt_imu_o, rbt_imu_a);
            #rx_1 = motion_model.x; ry_1 = motion_model.y; ro_1 = motion_model.o
#            rx = (rx*0.2 + rx_1 *0.8)
  #          ry = (ry *0.2+ rx_1*0.8)
            #ro = (ro*0.2 + ro_1*0.8)
            #rbt_pos = motion_model.calculate(rbt_wheels) # see next line

            #lab3_combi
            # ~ exact rbt index  
            
#            rbt_pos = (rx,ry,ro)
            rbt_pos_x = rx; rbt_pos_y = ry 
            # exact robot index


            #riE = rbt_pos[0] ;  rjE = rbt_pos[1]
            # ~ robot index using int and round. 
            # ~ int64 and float64 are slower than int and float respectively
            #ri = int(round(riE)); rj = int(round(rjE))            
            
            # increment update iteration number
            #update += 1
            # for each degree in the scan
            '''
            for i in xrange(360):
                if scan[i] != inf: # range reading is < max range ==> occupied
                    end_pos = inverse_sensor_model(scan[i], i, rbt_pos)
                    # set the obstacle cell as occupied
                    occ_grid.update_at_pos(end_pos, True)
                else: # range reading is inf ==> no obstacle found
                    end_pos = inverse_sensor_model(MAX_RNG, i, rbt_pos)
                    # set the last cell as free
                    occ_grid.update_at_pos(end_pos, False)
                # set all cells between current cell and last cell as free
                for idx in los.calculate(rbt_pos, end_pos):
                    occ_grid.update_at_idx(idx, False)
                    occ_grid.update_at_idx(idx, False)
            ''' 

            for i in xrange(360):        
                if scan[i] != inf: # range reading is < max range ==> occupied
                    #eiE = x2iE(rx + scan[o] * cos(ro + DEG2RAD[o])); ejE = y2jE(ry + scan[o] * sin(ro + DEG2RAD[o]))
                    end_pos_x = rx + scan[i] * cos(ro + DEG2RAD[i])
                    end_pos_y = ry + scan[i] * sin(ro + DEG2RAD[i])
                    #end_idx = pos2idx((end_pos_x,end_pos_y))
                    end_idx_i = x2i(end_pos_x); end_idx_j = y2j(end_pos_y)
                    # set the obstacle cell as occupied
                    update_at_idx(end_idx_i , end_idx_j, True) 
    
                else: # range reading is inf ==> no obstacle found
                    end_pos_x = rx + MAX_RNG * cos(ro + DEG2RAD[i])
                    end_pos_y = ry + MAX_RNG * sin(ro + DEG2RAD[i])

                    #end_idx = pos2idx((end_pos_x,end_pos_y))
                    end_idx_i = x2i(end_pos_x); end_idx_j = y2j(end_pos_y)
                    # set the last cell as free
                    update_at_idx(end_idx_i , end_idx_j, False) 
                    #occ_grid.update_at_idx(end_pos, False)
                # set all cells between current cell and last cell as free
                for idx in los_calc(rbt_pos_x,rbt_pos_y, end_pos_x, end_pos_y): #feed in exact idx instead of round
                    update_at_idx(idx[0] , idx[1], False)
                    
            # plan
            #rbt_idx = pos2idx(rbt_pos)
            rbt_idx_i = x2i(rbt_pos_x); rbt_idx_j = y2j(rbt_pos_y)
            #goal_idx = pos2idx(goal_pos)
            
              # plan
            # ~ update_at_idx will signal a path replanning via occ_grid.need_path if the existing path (see next) lies on an inflated or obstacle cell
            # ~ update_at_idx checks against an internally stored path updated using occ_grid.update_path(p), 
            # ~ where p is the new path to update that is returned from the post-process or A*
            # ~ occ_grid.show_map also uses the internally stored path to draw the path in image
            # ~ methods below are not cached for clarity
            if occ_grid.need_path == 1:   
            
                print ("path is replanned" )

                path = planner_get_path(rbt_idx_i,rbt_idx_j, goal_idx_i,goal_idx_j)
                if not path: #if path is empty use path_old to restore path
                    print("path is empty, using path_old ")
                    path = occ_grid.path_old     
                path, pts = post_processes(path,  occ_grid,los_int, los_pp) 
                #print ("path is " , path)

                #lab3_combi start use pts if path_process working
                p = len(pts)
                #p = len(path)
    
                if p <= 1:
                    # send shutdown
                    msg_main[0] = 1.
                    publisher_main.publish(msg_m)
                    # wait for sometime for move node to pick up mess age
                    t += 0.3
                    while rospy.get_time() < t:
                        pass
                    break
                # ~ update the internally stored path in occ_grid
                occ_grid.update_path(path)
                # ~ reset the occ_grid.need_path
                occ_grid.need_path = 0
                # ~ get the pt. number, starting from the pt. after the rbt idx, for robot to pursue
                p= len(pts) -2 
                #print (" p is " , p)        
                #px , py = idx2pos( ((path[p][0]), (path[p][1])))
                #px , py = idx2pos( (( pts[p][0]), (pts[p][1])))
                px = i2x(pts[p][0]) ; py = j2y(pts[p][1])
                #print (" px and py are" , px , py)
            
            
            dx = px-rx; dy = py-ry            
            err_pos = sqrt(dx*dx + dy*dy)
            
            if err_pos < NEAR_RADIUS: # 
                p -= 1
                if p < 0: # move to next goal, no more pts
                    # signal that a new path is needed
                    occ_grid.need_path = 1
                    g += 1
                    if g >= g_len: # no more goals
                        # send shutdown
                        msg_main[0] = 1.
                        publisher_main.publish(msg_m)
                        # wait for sometime for move node to pick up message
                        t += 0.3
                        while rospy.get_time() < t:
                            pass
                        break
                    #print("found new goal")        
                    # get the next goal pos for replanning
                    gx = goals[g][0]; gy = goals[g][1]
                    #goal_pos = (gx,gy)
                    #goal_idx = pos2idx(goal_pos) 
                    goal_idx_i = x2i(gx); goal_idx_j = y2j(gy)
                else: #
                    #px , py  =  idx2pos( ((path[p][0]), (path[p][1])))
                    #px , py = idx2pos( ((int (pts[p][0])), (int(pts[p][1]))))
                    px = i2x(int(pts[p][0])) ; py = j2y(int(pts[p][1]))
    
                    
                    
                    

    # number of goals (i.e. areas)
    # get the first goal idx
    #gi = occ_grid.x2i(gx); gj = occ_grid.y2j(gy)

    # set the goal number as zero
    
    
            # prepare message for sending
            print (goal_idx_i , goal_idx_j)
            msg_main[1] = px
            msg_main[2] = py
            publisher_main.publish(msg_m)
#lab3_combi end
            #print("path process.." ,path_process)
            #print("path process len" , len(path_process))
            
            #print("path" , path)
            
            # show the map as a picture
            show_map(rbt_idx_i , rbt_idx_j, path, goal_idx_i ,goal_idx_j)
#          show_map(rbt_idx, path_process, goal_idx)
            
            # increment the time counter
            et = rospy.get_time() - t
            print('[INFO] MAIN ({}, {:.3f})'.format(et <= 0.2, et)) #lab3_combi
            t += 0.2
            
#            print occ_grid.idx2cell((1,2)).is_occupied()
            

        
'''        
if __name__ == '__main__':      
    try: 
        main()
    except rospy.ROSInterruptException:
        pass
'''
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
        
        start_pose = sys.argv[5]
        start_pose = start_pose.split(',')
        start_pose = (float(start_pose[0]), float(start_pose[1]), 0.)
        main(goals, cell_size, min_pos, max_pos , start_pose)
    except rospy.ROSInterruptException:
        pass

