#!/usr/bin/env python3
import numpy as np
from matplotlib import pyplot as plt
import numba as nb
import time
import matplotlib.animation as animation

# Tridiagonal solver to be used by ADI routines
@nb.njit((nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:]))
def tridiagonalSolver(sub,dia,sup,rhs,xtmp):
	b = np.copy(dia)
	d = np.copy(rhs)
	ni = rhs.shape[0]
	for i in range(1,ni) :
		w = sub[i]/b[i-1]
		b[i] = b[i] - w*sup[i-1]
		d[i] = d[i] - w*d[i-1]
	xtmp[-1] = d[-1]/b[-1]
	for i in range(ni-2,-1,-1) :
		xtmp[i] = (d[i] - sup[i]*xtmp[i+1])/b[i]
	return

# ADI Methods for the unsteady heat equation

# Implicit i-direction, sweep across i-planes
# Check for ndims for 1 or 2 dimensional algorithm
# Inputs:
#	tridiagonal coefs, rhs array, ndims
# Outputs: Intermediate (or final) solution DU (AS REFERENCE)

@nb.njit
def isweep(sub,dia,sup,rhs,ndims,DU):
	# 1 DIMEMENSIONAL ALGORITHM
	tridiagonalSolver(sub,dia,sup,rhs,DU)
	return

@nb.njit
def isweep_2D(sub,dia,sup,rhs,ndims,DU):
	# 2 DIMENSIONAL ALGORITHM
	for j in range(1,rhs.shape[1]-1):
		tridiagonalSolver(sub,dia,sup,rhs[:,j],DU[:,j])
	return

@nb.njit
def isweep_3D(sub,dia,sup,rhs,ndims,DU):
	# 3 DIMENSIONAL ALGORITHM
	for j in range(1,rhs.shape[1]-1):
		for k in range(1,rhs.shape[2]-1):
			tridiagonalSolver(sub,dia,sup,rhs[:,j,k],DU[:,j,k])
	return
# j-direction implicit -- sweep across j-planes
@nb.njit
def jsweep_2D(sub,dia,sup,rhs,ndims,DU):
	# 2 DIMENSIONAL ALGORITHM
	for i in range(1,rhs.shape[0]-1):
		tridiagonalSolver(sub,dia,sup,rhs[i,:],DU[i,:])
	return

@nb.njit
def jsweep_3D(sub,dia,sup,rhs,ndims,DU):
	# 3 DIMENSIONAL ALGORITHM
	for i in range(1,rhs.shape[0]-1):
		for k in range(1,rhs.shape[2]-1):
			tridiagonalSolver(sub,dia,sup,rhs[i,:,k],DU[i,:,k])
	return
# k-direction implicit -- sweep across k-planes
@nb.njit
def ksweep(sub,dia,sup,rhs,DU):
	for i in range(1,rhs.shape[0]-1):
		for j in range(1,rhs.shape[1]-1):
			tridiagonalSolver(sub,dia,sup,rhs[i,j,:],DU[i,j,:])
	return

# Solver functions for heat equation
@nb.njit(nb.float64[:,:](nb.float64[:],nb.float64[:],nb.int64,nb.int64,nb.float64,nb.float64,nb.int64[:],nb.float64[:]))
def HeatEquation1D(x,u0,iters,nsave,dt,alpha,btypes,bvals):
	ndims = 1
	ni = x.shape[0]
	nt = iters
	dxs = x[1]-x[0]
	dxe = x[-1]-x[-2]
	isBoundaryType = btypes[0]
	ieBoundaryType = btypes[1]
	isBoundaryValue = bvals[0]
	ieBoundaryValue = bvals[1]

	w1 = 0.0
	w2 = 0.0

	if isBoundaryType == 1 :
		isBoundaryValue = isBoundaryValue*dxs
		w1 = 1.0
	if ieBoundaryType == 1 :
		ieBoundaryValue = ieBoundaryValue*dxe
		w2 = 1.0
	
	# Allocate array for storage
	u = np.zeros(ni)
	usave = np.zeros((ni,np.int64(iters/nsave)))
	# Apply boundary conditions
	u0[0]  = w1*u0[1]  + isBoundaryValue	
	u0[-1] = w2*u0[-2] + ieBoundaryValue	
	u[0] = u0[0]
	u[-1] = u0[-1]
	u[:] = u0[:]
	# Generate arrays tridiagonal solver (passed through isweep)
	sub = np.zeros(ni)
	dia = np.ones(ni)
	sup = np.zeros(ni)
	
	sub[:] = -dt*alpha/dxs**2
	dia[:] = 1.0 + 2.0*alpha*dt/dxs**2
	sup[:] = -dt*alpha/dxs**2
	sub[-1] = 0.0
	dia[0] = 1.0
	dia[-1] = 1.0
	sup[0] = 0.0
	
	for n in range(nt):	
		# Update boundary conditions
		u0[0]  = w1*u[1]  + isBoundaryValue	
		u0[-1] = w2*u[-2] + ieBoundaryValue	
		u[0]   = u0[0]
		u[-1]  = u0[-1]
		isweep(sub,dia,sup,u0,ndims,u)
		u0[:] = u[:]
		if np.mod(n,nsave) == 0 :
			ind = np.int64(n/nsave)
			usave[:,ind] = u[:]
	return usave

def oneDimensionalExample():
	ni = 50
	x = np.linspace(0,2.0*np.pi,ni)	
	u0 = np.zeros(ni)
	u0 = 5.0 + 5.0*np.sin(x)
	iters = 6000
	nsave = 50
	dt = 0.0001
	alpha = 12.5
	btypes = np.zeros(2,dtype=np.int64)
	bvals  = np.zeros(2,dtype=np.float64)
	btypes[0] = 1
	btypes[1] = 1
	bvals[0] =  0.0
	bvals[1] =  0.0
	
	tic = time.time()
	u  = HeatEquation1D(x,u0,iters,nsave,dt,alpha,btypes,bvals)
	u0 = 5.0 + 5.0*np.sin(x)
	btypes[0] = 1
	btypes[1] = 1
	bvals[0] =  0.0
	bvals[1] =  0.0
	u2 = HeatEquation1D(x,u0,iters,nsave,dt,alpha*0.5,btypes,bvals)
	toc = time.time()
	print("Elapsed time is ", toc-tic)
	
	xx = np.zeros(ni)
	xx[:] = x[:]
	y = 5.0+5.0*np.sin(x) 	
	plt.plot(x,u[:,100],'r-',label="alpha = 12")
	plt.plot(x,u2[:,100],'go',label="alpha = 6")
	plt.plot(xx,y,'ko',label="Initial condition")
#	plt.xlim((-2, 12))
	plt.ylim((-0.5, 10.5))
	plt.grid(True)
	plt.legend()
	plt.title('1D Heat Equation with Perfectly Insulated Ends (tau = 0.5)')
	plt.show()

def main():
	oneDimensionalExample()

if __name__ == "__main__":
	main()	








