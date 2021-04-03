from classes.room import room
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
from pprint import pprint

#This is how a room is initialzed
#N is how many parts we want to divide a room into in the x and y axis
n = 20
#The arguments 1,1 are the dimension of the room
r1 = room(1,1,n)
#The top temperature is set to 15 (wall)
r1.set_top(0,0,15)
#The bottom temperature is set to 15 (wall)
r1.set_bottom(0,0,15)
#The left temperature is set to 40 (heater)
r1.set_left(0,0,40)
#The right temperature is set to 15 (heater)
r1.set_right(0,0,15)
#All points with temp 0 are set to the average of the room
r1.fill_v()
#The right wall is set to be a boundry with a neumann conditon
r1.add_room_boundary(0,0,"right", "neumann")



#The dimensions are now 1x2 and for each e.g. set_right() which part of the room must be specified
r2 = room(1,2,n)
r2.set_top(0,0,40)
r2.set_left(0,0,15)
r2.set_right(0,0,15)
r2.add_room_boundary(0,0, "right", "dirichlet")
r2.set_right(0,1,15)
r2.set_bottom(0,1,5)
r2.set_left(0,1,15)
r2.add_room_boundary(0,1,"left", "dirichlet")
r2.fill_v()




r3 = room(1,1,n)
r3.set_top(0,0,15)
r3.set_right(0,0,40)
r3.set_bottom(0,0,15)
r3.set_left(0,0,15)
r3.add_room_boundary(0,0,"left", "neumann")
r3.fill_v()

#Here we do the parallel part
#Calling r1(), r2()... solved the conditons given in that room
#Some of the indexing is not dynamic and wouldn't work for rooms of other sizes.
#We use three processors

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
w = 0.8

for i in range(40):

    if rank == 0:

        r2_old = r2.v
        r2()
        v_matrix = r2.v.reshape(r2.steps_y, r2.steps_x)
        r1_beyond = v_matrix[n+1:-1,1] - v_matrix[n+1:-1,0]
        r3_beyond = v_matrix[1:n-1,-2] - v_matrix[1:n-1,-1]
        comm.send(r1_beyond, dest=1 )
        comm.send(r3_beyond, dest=2)
        b1 = comm.recv(source=1)
        b3 = comm.recv(source=2)
        r2.v[n*(n+1): n*n*2-n: n] = b1
        r2.v[2*n-1:n*n-1:n] = b3
        r2.v = w*r2.v + (1-w)*r2_old



    if rank == 1:
        r1_old = r1.v
        r1_beyond = comm.recv(source=0 )
        r1(r1_beyond)
        b1 = r1.v[r1.boundary[1:-1]]
        comm.send(b1, dest=0)
        r1.v = w*r1.v + (1-w)*r1_old


    if rank == 2:
        r3_old = r3.v
        r3_beyond = comm.recv(source=0 )
        r3(r3_beyond)
        b3 = r3.v[r3.boundary[1:-1]]
        comm.send(b3, dest=0)
        r3.v = w*r3.v + (1-w)*r3_old

#This part plots the room nicely, it woudln't work for rooms of any other dimension
if rank == 1:
    r1_matrix = r1.v.reshape(r1.steps_y, r1.steps_x)
    comm.send(r1_matrix, dest=0)

if rank == 2:
    r3_matrix = r3.v.reshape(r3.steps_y, r3.steps_x)
    comm.send(r3_matrix, dest=0)

if rank == 0:
    r1_matrix = comm.recv(source=1)
    r2_matrix = r2.v.reshape(r2.steps_y, r2.steps_x)
    r3_matrix = comm.recv(source=2)
    apartment=np.zeros([40,60])
    apartment[20:40,0:20] = r1_matrix
    apartment[0:40,20:40] = r2_matrix
    apartment[0:20,40:60] = r3_matrix
    value = 0
    masked_array = np.ma.masked_where(apartment == value, apartment)

    cmap = matplotlib.cm.spring  # Can be any colormap that you want after the cm
    cmap.set_bad(color='white')
    plt.imshow(masked_array, cmap="coolwarm")

    plt.show()
