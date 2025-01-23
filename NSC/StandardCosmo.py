import numpy as np#you usually need numpy


# load the modules
import os


from sys import path as sysPath
from sys import argv as argv


import pandas as pd
from scipy import interpolate

base_dir = os.path.dirname(os.path.realpath(__file__))



gTab = pd.read_table(base_dir+"/Data_gstars/gstar.dat",  names=['T','gstar'])

Ttab = gTab.iloc[:,0]
gtab = gTab.iloc[:,1]
tck  = interpolate.splrep(Ttab, gtab, s=0)

def gstar(T):
    
    condlist = [ T > 1e6 , T<=1e6]
    funclist = [interpolate.splev(1e6, tck, der=0), interpolate.splev(T, tck, der=0)] 
    return np.piecewise(T, condlist, funclist)

def dgstardT(T):
    condlist = [ T > 1e6 , T<=1e6]
    funclist = [interpolate.splev(1e6, tck, der = 1), interpolate.splev(T, tck, der = 1)]     
    
    return np.piecewise(T, condlist, funclist)

gSTab = pd.read_table(base_dir+"/Data_gstars/gstarS.dat",  names=['T','gstarS'])

TStab = gSTab.iloc[:,0]
gstab = gSTab.iloc[:,1]
tckS  = interpolate.splrep(TStab, gstab, s=0)

def gstarS(T):

    condlist = [ T > 1e6 , T<=1e6]
    funclist = [interpolate.splev(1e6, tckS, der = 0), interpolate.splev(T, tckS, der = 0)]  
    return np.piecewise(T, condlist, funclist)

def dgstarSdT(T):

    condlist = [ T > 1e6 , T<=1e6]
    funclist = [interpolate.splev(1e6, tckS, der = 1), interpolate.splev(T, tckS, der = 1)]  
    return np.piecewise(T, condlist, funclist)


GCF   = 6.70883e-39      # Gravitational constant in GeV^-2

MPl = 2.435e18



def rhoR(Temp):
    
    return (np.pi**2/30)*np.vectorize(gstar)(Temp)*Temp**4

def HubRad(Temp):
    Hsqr = (np.pi**2 / 90 )*np.vectorize(gstar)(Temp)* Temp**4/MPl**2
    return np.sqrt(Hsqr)

def sRAD(Temp):
    return ((2*np.pi**2)/(45))*np.vectorize(gstarS)(Temp)*Temp**3



