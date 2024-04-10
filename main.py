import numpy as np
import matplotlib.pyplot as plt

#This block of code defines the computational domain and generates mesh
length_x = 1000   #This is the width of the chip
length_y = 100    #This is the height of the chip

nx = 125   #This is the number of divisions in the x-axis
ny = 10    #This is the number of divisions in the y-axis 
 
a_x = np.linspace(0, length_x, nx)
a_y = np.linspace(0, length_y, ny)

dx = int(length_x/nx)  #This is the side of the grid element in the x-axis
dy = int(length_y/ny) #This is the side of the grid element in the y-axis

X, Y = np.meshgrid(a_x, a_y) #Creates the mesh for the computational domain

#This block of code defines the boundary conditions at the four side walls of the computational domain
Ttop = 20
Tbottom = 70
Tleft = 20
Tright = 20

#This generates an initial guess value of 0 for the internal grid points 
Tguess = 0

#Create an i,j matrix that contains the T values 
T = np.zeros((ny,nx))


T.fill(Tguess) #This fills the guess value into the T matrix 

#This defines the Q and k value of each material. You can change the material by removing the comment symbol (#) for copper and boron arsenide
Q = 3.0e-5 #This is the value of internal heat generation
k = 1.48e-4 #This is the value of thermal conductivity for silicon
#k = 1.3e-3 #This is the value of thermal conductivity for cubic boron arsenide
#k = 3.98e-4 #This is the value of thermal conductivity for copper


#The f value represents the forcing function for Poisson (in this case Q/k)
f =np.zeros_like(T) #creates a similar matrix to T with same dimension
f.fill(Q/k)

#initial conditions (Note the type of boundary condition)

T[0, :] = Tbottom  #row 0, all columns are given values
T[(ny-1),:] = Ttop #Topmost row all colums are given values
T[:,0] = Tleft #leftmost side zero column all rows are given value
T[:,(nx-1)] = Tright #Rightmost side column, all rows are given valen
np.set_printoptions(edgeitems=100)

#Now let's start with our numerical solution

beta = dx/dy #This is the grid aspect ratio
denom = (1/(2*(1 + beta**2)))
w = 1.5 #This is relaxation factor. It affects the rate of convergence

#Successive over relaxation method was used to solve the system of linear equations
error = np.ones_like(T)
for iteration in range (0, 500):
    T_old = T.copy()
    for i in range (1, (ny-1)): 
        for j in range (1, (nx-1)): 
            T[i,j] =(1-w)*T_old[i,j] + w*denom*(T[i+1][j] + T[i-1][j]+(beta**2)*T[i][j+1]+(beta**2)*T[i][j-1]+(dx**2)*f[i,j])           
            error[i,j] = abs(T[i,j] -T_old[i,j]) 
    if np.allclose(T[1:-1,1:-1], T_old[1:-1,1:-1], atol = 1e-6):  #Convergence criteria. The boundary values are fixed so the error considered was for inner elements
        break
    T_old = T 
    
print("The number of iterations needed for convergence is: ", iteration)
print ("The maximum temperature at the mid-plane of computational domain is : ", T[(ny-5),:].max())
np.set_printoptions(edgeitems=100)

#This code plots the upper temperature distribution at a plane around the lower surface, mid-surface and top surface of the domain

width = np.arange(0,1000, 8)
plt.plot(width, T[(ny-2),:].T,'r', label =" Temperature distribution at approximately topmost surface of the chip")
plt.plot(width, T[(ny-5),:].T,'b', label ="Temperature distribution at midsplane of the chip")
plt.legend(loc = 'lower right', fontsize = 8)
plt.xlabel("Length of the VLSI chip (um)")
plt.ylabel("Temperature (degrees C)")

#This creates the contour plots and colorbar
colorinterpolation = 50
colourMap = plt.cm.jet
fig, ax = plt.subplots(figsize=(10,1))
ax.set_aspect('equal')
cf = ax.contourf(X,Y,T, colorinterpolation, cmap = colourMap)
fig.colorbar(cf, ax=ax, label = 'Temperature distribution (C)', orientation = 'horizontal')
plt.show()
