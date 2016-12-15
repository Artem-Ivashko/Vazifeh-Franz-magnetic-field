#!/usr/bin/env python

from __future__ import division

# kwant
import kwant

# standard imports
import numpy as np
from scipy.sparse import linalg as lag
from matplotlib import pyplot as plt

from tinyarray import array as ta
from numpy import kron, sign, sqrt, fabs, sin, cos

# Pauli matrices
sigma0 = ta([[1, 0], [0, 1]])
sigma1 = ta([[0, 1], [1, 0]])
sigma2 = ta([[0, -1j], [1j, 0]])
sigma3 = ta([[1, 0], [0, -1]])

# products of Pauli matrices
s0s0 = kron( sigma0,sigma0 )
s0s1 = kron( sigma0,sigma1 )
s0s2 = kron( sigma0,sigma2 )
s0s3 = kron( sigma0,sigma3 )

s1s0 = kron( sigma1,sigma0 )
s1s1 = kron( sigma1,sigma1 )
s1s2 = kron( sigma1,sigma2 )
s1s3 = kron( sigma1,sigma3 )

s2s0 = kron( sigma2,sigma0 )
s2s1 = kron( sigma2,sigma1 )
s2s2 = kron( sigma2,sigma2 )
s2s3 = kron( sigma2,sigma3 )

s3s0 = kron( sigma3,sigma0 )
s3s1 = kron( sigma3,sigma1 )
s3s2 = kron( sigma3,sigma2 )
s3s3 = kron( sigma3,sigma3 )



# Simple Namespace
class SimpleNamespace(object):
    """A simple container for parameters."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs) 
       
       
        
def LandauEnergyTh( LLNumber, Parameters, Deltapz = 0., NodeNumber = 1 ):
    if NodeNumber == 1:
        EnergyNode = Parameters.EnergyNode1
        VelocityZNode = Parameters.VelocityZNode1
        VelocityXNode = Parameters.VelocityXNode1
    else:
        EnergyNode = Parameters.EnergyNode2
        VelocityZNode = Parameters.VelocityZNode2
        VelocityXNode = Parameters.VelocityXNode2
        
    tempVar = (Deltapz**2)*(VelocityZNode**2) + 2.*fabs(LLNumber)*Parameters.lBinv2*(VelocityXNode**2)
    if LLNumber == 0:
        return EnergyNode
    else:
        return EnergyNode + sign(LLNumber)*sqrt(tempVar)
    
    
    
def onsite_1D( site,p ):
    # for magnetic field in z-direction, Landau gauge
    x,=site.pos
    pyLong = p.py - float(x-p.x_shift)*float(p.lBinv2)
    
#It is the on-site energy according to the Hamiltonian that is written in Baireuther et al.'16
#And that seems to be actually realized in the Paul's code "franz_model.py"
    return p.tp*sin(pyLong)*s3s2 + p.tzp*sin(p.pz)*s2s0 \
    + ( p.M0+p.t+p.t*(1-cos(pyLong))+p.tz*(1-cos(p.pz)) )*s1s0 \
    + p.b0/2.*s2s3 + p.betaz/2.*s0s3
#And below is the on-site energy implemented in the code that was given to Artem by Paul on 28-Nov'16
#Which seems to give the same spectrum. The differences are in some of the 4x4 matrices
#    return   p.tp*sin(pyLong)*s3s2 + p.tzp*sin(p.pz)*s3s3 \
#             + ( p.M0+p.t+p.t*(1-cos(pyLong))+p.tz*(1-cos(p.pz)))*s1s0 \
#             + p.b0/2.*s3s0 + p.betaz/2.*s0s3

        

def hop_1D( s1,s2,p ):
#The hopping according to the Hamiltonian that is written in Baireuther et al.'16
#is the same it was implemented in the code that was given to Artem by Paul on 28-Nov'16
    return - 0.5j*p.tp*s3s1 - 0.5*p.t*s1s0



def FinalizedSystem_1D( SitesCount_X ):
    # lattices
    lat = kwant.lattice.general( ( (1,), ) )
    # builder
    sys = kwant.Builder()
    # first, define all sites and their onsite energies
    for nx in range(SitesCount_X):
        sys[lat(nx,)]=onsite_1D
    # hoppings
    for nx in range(SitesCount_X-1):
        sys[lat(nx+1,),lat(nx,)]=hop_1D    
    # finalize the system
    return sys.finalized()
    
    
    
def diagonalize_1D( FinalizedSystem, Parameters ):
    ham_sparse_coo = FinalizedSystem.hamiltonian_submatrix( args=([Parameters]), sparse=True )
    #Conversion to some "compressed sparse" format
    ham_sparse = ham_sparse_coo.tocsc()
    #Finding k=EigenvectorsCount eigenvalues of the Hamiltonian H, which are located around sigma=0, 
    #so that they are the closest (wrt to the Hermitian norm) to this value. (The largest-in-magnitude label ('LM') is misleading,
    #since here we work in the "shift-invert" mode, so that actually the largest eigenvalues of the 1 / (H - omega) matrix 
    #are sought for. The numerical check is in agreement with this picture.)
    EigenValues, EigenVectors = lag.eigsh( ham_sparse,k=Parameters.EigenvectorsCount,return_eigenvectors=True, \
                                          which='LM',sigma=Parameters.FermiEnergy, tol=Parameters.EnergyPrecision )
    EigenVectors = np.transpose(EigenVectors)
    #Sorting the wavefunctions by eigenvalues, so that the states with the lowest energies come first
    idx = EigenValues.argsort()
    EigenValues = EigenValues[idx]
    EigenVectors = EigenVectors[idx]
    
    SitesCount_X = len(FinalizedSystem.sites)
    EigenVectors = np.reshape(EigenVectors, (Parameters.EigenvectorsCount, SitesCount_X, Parameters.WavefunctionComponents) )
    return EigenValues, EigenVectors



def pSweep_1D( FinalizedSystem, Parameters, pMin, pMax, pCount, yORzSweep ):
    pSweep = np.linspace(pMin, pMax, pCount)
    EigenValuesSweep = []
    EigenVectorsSweep = []
    for p in pSweep:
        if yORzSweep == 'pzSweep':
            Parameters.pz = p
        elif yORzSweep == 'pySweep':
            Parameters.py = p
        else:
            raise ValueError("Only two values are possible for the parameter yORzSweep: either 'pzSweep' or 'pySweep'")
        
        EigenValues, EigenVectors = diagonalize_1D( FinalizedSystem, Parameters )
        EigenValuesSweep.append(EigenValues)
        EigenVectorsSweep.append(EigenVectors)
    
    return EigenValuesSweep, EigenVectorsSweep



#The procedure plots the number density as a function of coordinate (horizontal axis)
#The sweep is over different energies (vertical axis, however the coordinate on that axis is not the energy itself, but rather an index of the energy)
def density_plot_1D( FinalizedSystem, Parameters, EigenVectors ):
    position_1D = realspace_position_1D(FinalizedSystem)
    SitesCount_X = len(FinalizedSystem.sites)    
    
    density = [[np.vdot(EigenVectors[i][position_1D[j]],EigenVectors[i][position_1D[j]]) for j in range(SitesCount_X)] \
               for i in range(Parameters.EigenvectorsCount)]
    density = np.real(density)

    plt.pcolor( np.linspace(0,SitesCount_X,SitesCount_X), np.linspace(0,1,Parameters.EigenvectorsCount), density, \
               vmin = 0, vmax= 3./SitesCount_X)
    plt.colorbar()
    plt.show()
    
    
    
def spectrum_plot_1D( EigenValues, pMin, pMax, pCount ):
    pSweep = np.linspace(pMin, pMax, pCount)
    plt.plot(pSweep, EigenValues,"k.",markersize=3)
    plt.xlim(pMin,pMax)
    plt.ylabel('Energy [t]')
    plt.show()
    
    

def realspace_position_1D( FinalizedSystem ):
    SitesCount_X = len(FinalizedSystem.sites)
    return [[j for j in range(SitesCount_X) if FinalizedSystem.sites[j].pos[0] == i][0] for i in range(SitesCount_X)]


