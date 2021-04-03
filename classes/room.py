
import numpy as np
import scipy.linalg as la
# import sparse module from SciPy package
from scipy import sparse
from .laplacian_solver import laplacian_solver
# import uniform module to create random numbers
from scipy.stats import uniform
import time
import math

"""
This class is used to represent one of the rooms. It is initialized by giving
the dimensions of the room rowsXcols and the number mesh_n that we want to divide each axis into
"""


class room:
    def __init__(self, rows, cols, mesh_n):
        self.rows = rows
        self.cols = cols
        self.mesh_n = mesh_n
        self.steps_x = rows*mesh_n
        self.steps_y = cols*mesh_n
        self.boundary = []
        self.condition = ""
        self.v = np.zeros(rows*cols*mesh_n*mesh_n)
        step_size = self.mesh_n
        self.outer_points = []

    """
    This method is used to solve a rooms dirichlet or neumann condition. For Dirlicht
    it simply updates the inner points in the roomself.
    For neumann the relevant inner points of the rooom with the Dirichlet conditons are passed alongself.
    The conditions, and their direction (if relevant) are saved in the class when defining boundries in
    add_room_boundary().
    """
    def __call__(self, beyond_points = None):
        v = self.v.reshape(self.steps_y, self.steps_x)
        top = v[0,1:self.steps_x-1]
        left = v[1:self.steps_y-1,0]
        right = v[1:self.steps_y-1,-1]
        bottom = v[-1,1:self.steps_x-1]
        b = np.zeros((self.steps_y-2, self.steps_x-2))
        b[0,:] +=  top
        b[:, 0] +=  left
        b[:, -1] += right
        b[-1, :] += bottom
        if(self.condition=="dirichlet"):
            #print(b.flatten())
            #print(self.omega(b.flatten()))
            self.v[self.get_inner_points()] = self.omega(b.flatten())

        elif(self.condition=="neumann"):
            b_direction = self.omega.b_direction
            b_new = beyond_points

            if(b_direction=="top"):
                b[0,:] -= top
                #b_new -= top
                b_new[0] += v[0,0]
                b_new[-1] += v[0,-1]

                b = np.vstack((b_new, b))
                v_new = self.omega.reconstruct(self.omega(b.flatten()))
                self.v[self.get_inner_points()] = v_new[1:,:].flatten()
                self.v[self.boundary[1:-1]] = v_new[0,:].flatten()

            elif(b_direction=="bottom"):
                b[-1, :] -= bottom
                #b_new -= bottom
                b_new[0] = v[-1,0]
                b_new[-1] += v[-1,-1]

                b = np.vstack((b, b_new))
                v_new = self.omega.reconstruct(self.omega(b.flatten()))

                self.v[self.get_inner_points()] = v_new[:-1,:].flatten()
                self.v[self.boundary[1:-1]] = v_new[-1,:].flatten()

            elif(b_direction=="left"):
                b[:, 0] -= left

                #b_new -= left
                b_new[0] += v[0,0]
                b_new[-1] += v[-1,0]
                b_new = b_new.reshape(len(b_new),1)


                b = np.hstack((b_new,b))

                v_new = self.omega.reconstruct(self.omega(b.flatten()))

                self.v[self.get_inner_points()] = v_new[:,1:].flatten()
                self.v[self.boundary[1:-1]] = v_new[:,1].flatten()

            elif(b_direction=="right"):
                b[:, -1] -= right

                #b_new -= right
                b_new[0] += v[0,-1]
                b_new[-1] += v[-1,-1]
                b_new = b_new.reshape(len(b_new),1)

                b = np.hstack((b, b_new))
                v_new = self.omega.reconstruct(self.omega(b.flatten()))

                self.v[self.get_inner_points()] = v_new[:,:-1].flatten()
                self.v[self.boundary[1:-1]] = v_new[:,-1].flatten()



    #Indexed from 0
    """
    These methods are used to set the temperatues at the boundries (walls, windows)
    If a room is larger than 1x1 you must also specify which part of the room, and which side.
    For the rest of the code to work correctly all boundries must be given a temperature, as a vector
    keeping track of the outer points is also defined and saved here
    """
    def set_left(self, row, col, temperature):
        offset = row*self.mesh_n + col*self.steps_x*self.mesh_n
        for i in range(self.mesh_n):
            index = offset + self.steps_x *i
            self.v[index] = temperature
            self.outer_points.append(index)

    def set_right(self, row, col, temperature):
        offset = self.mesh_n -1 + row*self.mesh_n + col*self.steps_x*self.mesh_n
        for i in range(self.mesh_n):
            index = offset + self.steps_x*i
            self.v[index] = temperature
            self.outer_points.append(index)

    def set_top(self, row, col, temperature):
        offset = self.mesh_n*self.steps_x*col + self.mesh_n*row
        for i in range(self.mesh_n):
            index = offset +i
            self.v[index] = temperature
            self.outer_points.append(index)

    def set_bottom(self, row, col, temperature):
        offset = (self.mesh_n-1)*self.steps_x + row*self.mesh_n + col*self.mesh_n*self.steps_x
        for i in range(self.mesh_n):
            index = offset + i
            self.v[index] = temperature
            self.outer_points.append(index)

    """This is where you add a room boundry. It saves which condition the boundry has
    and also which indices in the boundry has. The laplacian_solver class saved as self.omega, which solves
    the neumann or dirichlet conditon is also defined here
    """
    def add_room_boundary(self, row, col, side, condition):
        boundary = np.zeros(self.mesh_n)
        if(side=="left"):
            offset = row*self.mesh_n + col*self.steps_x*self.mesh_n
            for i in range(self.mesh_n):
                index = offset + self.steps_x *i
                boundary[i] = index
                self.outer_points.append(index)

        elif(side=="right"):
            offset = self.mesh_n -1 + row*self.mesh_n + col*self.steps_x*self.mesh_n
            for i in range(self.mesh_n):
                index = offset + self.steps_x*i
                boundary[i] = index
                self.outer_points.append(index)


        elif(side=="top"):
            offset = self.mesh_n*self.steps_x*col + self.mesh_n*row
            for i in range(self.mesh_n):
                index = offset + i
                boundary[i] = index
                self.outer_points.append(index)

        elif(side=="bottom"):
            offset = (self.mesh_n-1)*self.steps_x + row*self.mesh_n + col*self.mesh_n*self.steps_x
            for i in range(self.mesh_n):
                index = offset + i
                boundary[i] = index
                self.outer_points.append(index)

        if(condition == "dirichlet"):

            self.omega =  laplacian_solver(self.steps_y-2, self.steps_x-2, condition)
            self.condition = "dirichlet"
        elif(condition == "neumann"):
            self.condition = "neumann"
            if(side == "left" or side == "right"):
                self.omega =  laplacian_solver(self.steps_y-2,self.steps_x-1, condition, side)
            elif(side == "bottom" or side == "top"):
                self.omega =  laplacian_solver(self.steps_y-1,self.steps_x-2, condition, side)
        self.boundary = boundary.astype(int)

    #This method is used to set all zero points in a room to the average temperature in the room
    def fill_v(self):
        m = np.mean(self.v)
        idx = (self.v == 0)
        self.v[idx] = m

    #Prints v as a matrix
    def print_v(self):
        new_v = self.v.reshape(self.steps_y, self.steps_x)
        print(new_v)
    #Returnes all the outer points(Walls, windows, heaters, boundries) for a given room
    def get_outer_points(self):
        return list(set(self.outer_points))
    #Returnes all the interior points for a room
    def get_inner_points(self):
        return list(set(range(len(self.v))) - set(self.outer_points))
