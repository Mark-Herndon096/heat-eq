#!/usr/bin/env python3
import numpy as np
from matplotlib import pyplot as plt
from numba import njit, int64, float64, types, int32
import time

@njit((float64[:],float64[:],float64[:],float64[:],float64[:],int64))
def tridiagonalSolver(a,bb,c,dd,u,ni) :
	b = np.copy(bb)
	d = np.copy(dd)
	for i in range(1,ni) :
		w = a[i]/b[i-1]
		b[i] = b[i] - w*c[i-1]
		d[i] = d[i] - w*d[i-1]
	u[-1] = d[-1]/b[-1]
	for i in range(ni-2,-1,-1) :
		u[i] = (d[i] - c[i]*u[i+1])/b[i]
	return

@njit(float64[:](int64,float64[:],int32[:],float64[:],float64,float64[:],float64,int64,int64,float64,float64))
def backward_Euler_solver(nLayers,x,ib,alp,dt,dxa,tspan,ibt1,ibt2,bval1,bval2) :
	ni = x.size
	u0 = np.ones(ni)
	u  = np.zeros(ni)
	if ibt1 == 0 :
		u0[0] = bval1
		Q1 = bval1*alp[0]/dxa[0]
		w1 = 0.0
	if ibt2 == 0 :
		u0[-1] = bval2
		Q2 = bval2*alp[-1]/dxa[-1]
		w2 = 0.0
	if ibt1 == 1 :
		Q1 = -bval1
		w1 = 0.0
	if ibt2 == 1 :
		Q2 = bval2
		w2 = 0.0
	u = u0
	# Neumann BC handled in routine
	nt = int(np.floor(tspan/dt))
	# Initialize arrays for tridiagonal solver
	a = np.zeros(ni)
	b = np.zeros(ni)
	c = np.zeros(ni)
	istart = 0
	for m in range(nLayers) :
		ie = ib[m]	
		a[istart:ie+1] = -dt*alp[m]/dxa[m]**2
		b[istart:ie+1] = 1.0 + 2.0*dt*alp[m]/dxa[m]**2
		c[istart:ie+1] = -dt*alp[m]/dxa[m]**2
		istart = ib[m]+1
	for m in range(nLayers-1) :
		ind = ib[m]
		a[ind] = 0
		b[ind] = 1
		c[ind] = 0
	b[0]  = 1
	c[0]  = 0
	b[-1] = 1
	a[-1] = 0
	if ibt1 == 1 :
		c[0] = -1
	if ibt2 == 1 :
		a[-1] = -1
	# Primary loop for time series . . .
	for n in range(nt) :
		# Appply boundary conditions
		u0[0]  = w1*u[1]  + (dxa[0]/alp[0])*Q1
		u0[-1] = w2*u[-2] + (dxa[-1]/alp[-1])*Q2
		# Handle interface (from t = t-\Delta t)
		for m in range(nLayers-1) :
			ind = ib[m]
			c1 = 1.0/(1.0+(dxa[m]*alp[m+1]/(dxa[m+1]*alp[m])))
			c2 = dxa[m]*alp[m+1]/(dxa[m+1]*alp[m])
			u0[ind] = c1*(u0[ind-1]+c2*u0[ind+1])
		tridiagonalSolver(a,b,c,u0,u,ni)
		u0[:] = u[:]
	return u

def setProperties(nLayers,*args) :
	# Thickness i args[0,nLayers-1]
	# Alpha in args[nLayers,2*nLayers-1]
	# dx in args[2*nLayers]
	# Cmax in args[-1]
	xspan = 0.0
	dx = args[2*nLayers]
	ib = np.zeros(nLayers,dtype=np.int32)
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
	Cmax = args[-1]
	dx2  = dx**2
	dt   = Cmax*dx2/np.amax(alp)
	return [x, ib, alp, dt, dxa]

# Solve "cube" with one layer (one material)
# With Dirichlet boundary conditions
# T = 5 @ x = 0 & T = 10 & x = t1 = 10.0
def solve_single_layer_cubei() :
	nLayers = 1
	t1 = 10.0
	a1 = 14.2
	dx = 0.05
	tspan = 5.0
	[x,ib,alp,dt,dxa] = setProperties(nLayers,t1,a1,dx,25)
	u = backward_Euler_solver(nLayers,x,ib,alp,dt,dxa,tspan,0,0,5.0,10.0)
	return [x, u]
# Solve "cube" with two layers (two materials)
# With Dirichlet boundary conditions
# T = 5 @ x = 0 & T = 10 & x = t1 = 10.0
# With material interface at x = 4
def solve_double_layer_cubei() :
	nLayers = 2
	t1 = 4.0
	t2 = 6.0
	a1 = 45.0
	a2 = 11.2
	dx = 0.05
	tspan = 5.0
	[x,ib,alp,dt,dxa] = setProperties(nLayers,t1,t2,a1,a2,dx,25)
	u = backward_Euler_solver(nLayers,x,ib,alp,dt,dxa,tspan,0,0,5.0,10.0)
	return [x, u]

# Solve "cube" with two layers (two materials)
# With Neumann boundary conditions
# dT/dx = 0 @ x = 0 & dT/dx = Q, Q=5 & x = 10.0
def solve_double_layer_cube_2i() :
	nLayers = 2
	t1 = 4.0
	t2 = 6.0
	a1 = 45.0
	a2 = 11.2
	dx = 0.05
	tspan = 5.0
	[x,ib,alp,dt,dxa] = setProperties(nLayers,t1,t2,a1,a2,dx,2)
	u = backward_Euler_solver(nLayers,x,ib,alp,dt,dxa,tspan,1,1,0.0,5.0)
	return [x, u]
## MAIN CODE FOR EXAMPLES
def main() :
	tic = time.time()
	[x1i, u1i] = solve_single_layer_cubei()
	[x2i, u2i] = solve_double_layer_cubei()
	[x3i, u3i] = solve_double_layer_cube_2i()
	toc = time.time()
	print("Elapsed time = ", toc-tic, "seconds")
	
	# Plot 3 examples on subplot
	plt.figure()
	plt.subplot(1,3,1)
	plt.style.use('seaborn-deep')
	plt.plot(x1i,u1i,"r-")
	plt.xlim((0, 10))
	plt.xlabel("x (length)")
	plt.ylabel("u (temperature)")
	plt.grid(True)
	plt.title("Dirichlet conditions (1 Layer)")
	
	plt.subplot(1,3,2)
	plt.style.use('seaborn-deep')
	plt.plot(x2i,u2i,"r-")
	plt.xlim((0, 10))
	plt.xlabel("x (length)")
	plt.ylabel("u (temperature)")
	plt.grid(True)
	plt.title("Dirichlet conditions (2 Layer)")
	
	plt.subplot(1,3,3)
	plt.style.use('seaborn-deep')
	plt.plot(x3i,u3i,"r-")
	plt.xlim((0, 10))
	plt.xlabel("x (length)")
	plt.ylabel("u (temperature)")
	plt.grid(True)
	plt.title("Neumann conditions (2 Layer)")
	
	plt.show()

if __name__ == '__main__' :
	main()
