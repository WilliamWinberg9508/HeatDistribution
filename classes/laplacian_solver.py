# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from scipy.sparse import diags as diags
from scipy.sparse.linalg import spsolve as spsolve


class laplacian_solver:

    def __init__(self, dim_row, dim_col, b_condition, b_direction = None):
        """
        Initialises a "room" for solving the laplacian equation. Creates an
        appropriate matrix (stored as self.A_matrix in csr-format) for solving
        the matrix equation Ax = -b.

        Parameters
        ----------
        dim_row : int
            Number of rows.
        dim_col : int
            Numer of columns.
        b_condition : string
            Specify which conditions to use.
        b_direction : string, required for neumann conditions
            Specify in which direction the conditions apply.

        """
        self.dim_row = dim_row
        self.dim_col = dim_col
        self.b_condition = b_condition
        self.b_direction = b_direction
        if b_condition == "dirichlet":
            self.A_matrix = self.generate_dirichlet_matrix()
        elif b_condition == "neumann":
            self.A_matrix = self.generate_neumann_matrix(b_direction)
        else:
            raise Exception("Must specify which boundary conditions to use!")

    def __call__(self, b):
        """
        Solves Ax = -b for a given vector b
        """
        return spsolve(self.A_matrix, -b)

    def generate_dirichlet_matrix(self):
        n, m = self.dim_row, self.dim_col
        N = n*m

        d_0 = -4. # main diagonal
        d_1 = np.array((([1.]*(m-1) + [0.])*n)) # super diagonal (and sub)
        d_2 = 1. # super super diagonal (and sub sub)

        data = [d_2, d_1, d_0, d_1, d_2]
        offsets = [-m, -1, 0, 1, m]

        A = diags(data, offsets, shape = (N, N), format = "csr")
        return A

    def generate_neumann_matrix(self, boundary):
        n, m = self.dim_row, self.dim_col
        N = n*m

        if boundary == "right":
            d_0 = np.array((([-4.]*(m-1) + [-3.])*n))
        elif boundary == "left":
            d_0 = np.array((([-3.] + [-4.]*(m-1))*n))
        elif boundary == ("up" or "top"):
            d_0 = np.array((([-3.]*m + [-4.]*(N-m))))
        elif boundary == ("down" or "bottom"):
            d_0 = np.array((([-4.]*(N-m) + [-3.]*m)))
        else:
            raise Exception("Must specify a valid direction for Neumann conditions!")

        d_1 = np.array((([1.]*(m-1) + [0.])*n)) # super diagonal (and sub)
        d_2 = 1. # super super diagonal (and sub sub)

        data = [d_2, d_1, d_0, d_1, d_2]
        offsets = [-m, -1, 0, 1, m]

        A = diags(data, offsets, shape = (N, N), format = "csr")
        return A

    def reconstruct(self, x):
        """
        Reshapes the vector x into its "unflattened" matrix form.
        """
        return x.reshape(self.dim_row, self.dim_col)

### Testing Cases
# Uncomment section for examples

# Dirichlet
"""
n, m = 12,6
room_1 = laplacian_solver(n,m, "dirichlet")
left, right, up, down = 15, 15, 40, 5 # boundary conditions

b = [up + left] + [up]*(m-2) + [up + right]
b+= ([left] + [0]*(m-2) + [right])*(n-2)
b+= [down + left] + [down]*(m-2) + [down + right]
b = np.array(b)

new_x = room_1(b)
print("For boundary conditions")
print(b.reshape(n,m))
print("solution was found to be")
print(new_x.reshape(n,m))
"""

# Neumann
"""
n, m = 4,5
h = 1/(n+1) # step size
room_2 = laplacian_solver(n,m, "neumann", "left")
left, right, up, down = h*9, 40, 15, 15 # boundary conditions

b = [up + left] + [up]*(m-2) + [up + right]
b+= ([left] + [0]*(m-2) + [right])*(n-2)
b+= [down + left] + [down]*(m-2) + [down + right]
b = np.array(b)

new_x = room_2(b)
print("For boundary conditions")
print(b.reshape(n,m))
print("solution was found to be")
print(new_x.reshape(n,m))
"""
