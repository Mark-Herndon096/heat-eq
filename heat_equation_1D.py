#!/usr/bin/env python3
import numpy as np
from matplotlib import pyplot as plt


# Interfaces between layers are treated as perfect interfaces
# (no contact resistance), where continuity between temperature
# and heat flux are enforced. 
# For an equal spaced grid, the interface point is treated as
# 
# u_i = (1+alphaL/alphaR)^{-1} * (u_{i-1} + (alphaR/alphaL)*u_{i+1})
# 
# where alphaL == thermal diffusivity of lower material and
# alphaR == thermal diffusivity of upper material across interface

# As this is an explicit method (Forward Euler) the time step
# is quite small so this can take a long time to run. Sorry!

# INPUT:
#	number of layers
#	thickness 1, thickness 2, ... , thickness n
#	alpha 1, alpha 2, ... , alpha 2
#	delta_x --> minimum spacing (uniform cartesian grid ... 1D here)
#	delta_x will be modified such that grid points always begin and
#	on a boundary (dx in one layer may be slightly different than another)
def setProperties(nLayers,*args) :
	# Thickness i args[0,nLayers-1]
	# Alpha in args[nLayers,2*nLayers-1]
	# dx in args[2*nLayers]
	xspan = 0.0
	dx = args[2*nLayers]
	ib = np.zeros(nLayers,dtype=int)
	alp = np.zeros(nLayers)
	dxa = np.zeros(nLayers)
	for n in range(nLayers) :
		ni = int(np.floor(args[n]/dx));
		if n == 0 :
			xtmp = np.linspace(0.0,args[n],ni)
			xOld = args[n] 
			dxa[0] = xtmp[2]-xtmp[1]
		else :
			xtmp = np.linspace(xOld,xOld+args[n],ni)
			xOld = xOld+args[n]
			dxa[n] = xtmp[2]-xtmp[1]
		if n == 0 :
			x = xtmp
			ib[0] = ni-1
		else :
			x = np.concatenate((x[0:-1],xtmp), axis=0)
			ib[n] = ib[n-1] + ni-1
		alp[n] = args[nLayers+n]
	# dt derived from CFL condition of minimum grid cell size
	# and maximum thermal diffusivity. Taking CFL = 0.5 to help
	# ensure (but not guarantee) stability for an explicitly 
	# integrated scheme
	Cmax = 0.5
	dx2  = dx**2
	dt   = Cmax*dx2/np.amax(alp)
		
	return [x, ib, alp, dt, dxa]
	
# Solve 1D heat equation via Forward Euler (explicit time integration)
# With second order in space finite-difference and continuous 
# heat flux at interfaces (assumes perfect contact)
# *args contains boundary condition information
# args[0] = start boundary condition type (0 Dirichlet, 1 Neumann)
# args[1] = end boundary condition type (0 Dirichlet, 1 Neumann)
# args[3] = Value for start B.C.
# args[4] = Value for end B.C.
# Dirichlet specifies temperature, Neumann specifies heat flux
# where a value of 0 means perfectly insulated
# UNITS MUST BE CONSISTENT AS THIS METHOD HAS NOT BEEN NONDIMENSIONALIZED
# ---> x,dx in [m], alpha in [m^2/s] (Usually listed in [mm^2/s]
# 
# alpha = k/(rho * c_p) [L^2/T] 
# where alpha is thermal diffusivity
# k is thermal conductivity 
# rho is the density
# c_p is the specific heat capacity
# For heat flux boundary conditions:
#	Need to divide Q by (rho*cp) prior to passing argument
#	to equation solver so that the condition is dimensionally consistent
#	Recall heat flux [Watt/m^2] --> [M/T^3] 
#	So you must keep dimensions consistent with how you scale your heat flux
#	e.g. "cube" dimensions in [m], thermal diffusivity in [m^2/s], 
#	Heat flux [Watt/m^2], Density in [kg/m^3], Heat capcity [J/(kg*K)], etc ...
def solve_1D_heat_equation(nLayers,x,ib,alp,dt,dxa,tspan,*args) :
	ni = x.size
	u0 = np.zeros(ni)
	u  = np.zeros(ni)
	if args[0] == 0 :
		u0[0] = args[2]
		Q1 = args[2]*alp[0]/dxa[0]
		w1 = 0.0
	if args[1] == 0 :
		u0[-1] = args[3]
		Q2 = args[3]*alp[-1]/dxa[-1]
		w2 = 0.0
	if args[0] == 1 :
		Q1 = args[2]
		w1 = 1.0
	if args[1] == 1 :
		Q2 = args[3]
		w2 = 1.0
	u = u0
	# Neumann BC handled in routine
	nt = int(np.floor(tspan/dt))
	# Primary loop for time series . . .
	for n in range(nt) :
		# Appply boundary conditions
		u[0]  = w1*u[1]  + (dxa[0]/alp[0])*Q1
		u[-1] = w2*u[-2] + (dxa[-1]/alp[-1])*Q2
		# Redundancy because ...
		u0[0]  = u[0]
		u0[-1] = u[-1]
		# Loop over layers (Internal)	
		ie = 0;
		for m in range(nLayers) :
			c1 = alp[m]*dt/(dxa[m]**2)
			istart = ie + 1
			iend = ib[m]
			for ii in range(istart,iend,1) :
				u[ii] = u0[ii] + c1*(u0[ii-1]-2.0*u0[ii]+u0[ii+1])
			ie = ib[m]

		# Handle interface 
		for m in range(nLayers-1) :
			ind = ib[m]
			c1 = 1.0/(1.0+(dxa[m]*alp[m+1]/(dxa[m+1]*alp[m])))
			c2 = dxa[m]*alp[m+1]/(dxa[m+1]*alp[m])
			u[ind] = c1*(u0[ind-1]+c2*u0[ind+1])
		u0 = u
	# Returns solution only at t = tspan
	# Can modify easily to save every x iterations and return in 2D array
	# with shape (nsolutions,ni) where nsolutions are the number of solutions
	# saved total. Do not recommend saving every time step...
	return u

# Solve "cube" with one layer (one material)
# With Dirichlet boundary conditions
# T = 5 @ x = 0 & T = 10 & x = t1 = 10.0
def solve_single_layer_cube() :
	nLayers = 1
	t1 = 10.0
	a1 = 14.2
	dx = 0.05
	tspan = 5.0
	[x,ib,alp,dt,dxa] = setProperties(nLayers,t1,a1,dx)
	u = solve_1D_heat_equation(nLayers,x,ib,alp,dt,dxa,tspan,0,0,5,10)
	return [x, u]
# Solve "cube" with two layers (two materials)
# With Dirichlet boundary conditions
# T = 5 @ x = 0 & T = 10 & x = t1 = 10.0
# With material interface at x = 4
def solve_double_layer_cube() :
	nLayers = 2
	t1 = 4.0
	t2 = 6.0
	a1 = 45.0
	a2 = 11.2
	dx = 0.05
	tspan = 5.0
	[x,ib,alp,dt,dxa] = setProperties(nLayers,t1,t2,a1,a2,dx)
	u = solve_1D_heat_equation(nLayers,x,ib,alp,dt,dxa,tspan,0,0,5,10)
	return [x, u]

# Solve "cube" with two layers (two materials)
# With Neumann boundary conditions
# dT/dx = 0 @ x = 0 & dT/dx = Q, Q=5 & x = 10.0
def solve_double_layer_cube_2() :
	nLayers = 2
	t1 = 4.0
	t2 = 6.0
	a1 = 45.0
	a2 = 11.2
	dx = 0.05
	tspan = 5.0
	[x,ib,alp,dt,dxa] = setProperties(nLayers,t1,t2,a1,a2,dx)
	u = solve_1D_heat_equation(nLayers,x,ib,alp,dt,dxa,tspan,1,1,0,5)
	return [x, u]
## MAIN CODE FOR EXAMPLES
def main() :
	[x1, u1] = solve_single_layer_cube()
	[x2, u2] = solve_double_layer_cube()
	[x3, u3] = solve_double_layer_cube_2()
	
	# Plot 3 examples on subplot
	plt.figure()
	plt.subplot(1,3,1)
	plt.plot(x1,u1,"k-")
	plt.xlim((0, 10))
	plt.xlabel("x (length)")
	plt.ylabel("u (temperature)")
	plt.grid(True)
	plt.title("Dirichlet conditions (1 Layer)")
	
	plt.subplot(1,3,2)
	plt.plot(x2,u2,"k-")
	plt.xlim((0, 10))
	plt.xlabel("x (length)")
	plt.ylabel("u (temperature)")
	plt.grid(True)
	plt.title("Dirichlet conditions (2 Layer)")
	
	plt.subplot(1,3,3)
	plt.plot(x3,u3,"k-")
	plt.xlim((0, 10))
	plt.xlabel("x (length)")
	plt.ylabel("u (temperature)")
	plt.grid(True)
	plt.title("Neumann conditions (2 Layer)")
	
	plt.show()

if __name__ == '__main__' :
	main()
