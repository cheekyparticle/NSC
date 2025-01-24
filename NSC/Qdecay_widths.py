import numpy as np#you usually need numpy


# load the modules


from sys import path as sysPath
from sys import argv as argv
from . import StandardCosmo as SC 




GCF   = 6.70883e-39      # Gravitational constant in GeV^-2

MPl = 2.435e18


def sigv(mQ):
    
    cf = 2/9 
    cg = 220/27
    nf = 3 
    alphas = 0.118
    
    return np.pi*alphas**2 * (cf * nf + cg)/(16*mQ**2)


def rhoQ_FO(mQ, Temp):
    
    gfacts = np.sqrt(SC.gstar(mQ)) * (SC.gstarS(Temp)/SC.gstarS(mQ))
    
    return ((np.pi)/(25*np.sqrt(90))) * ((Temp**3)/(sigv(mQ) * MPl)) * gfacts


def rhoR(Temp):
    
    return (np.pi**2/30)*SC.gstar(Temp)*Temp**4




def tend_func(mQ, dim=6):
    
    TEND=0.0
    
    if dim == 6 : 
        Tguess = 2.6e-2 * (mQ/5e11)**(5/2) * (1/SC.gstar(mQ))**(1/4)
        
        TEND = 2.6e-2 * (mQ/5e11)**(5/2) * (1/SC.gstar(Tguess))**(1/4)
        
        
    if dim == 5:
        Tguess = 1.1e7 * (mQ/5e11)**(3/2) * (1/SC.gstar(mQ))**(1/4)
        
        TEND = 1.1e7 * (mQ/5e11)**(3/2) * (1/SC.gstar(Tguess))**(1/4)
        
    if dim == 7:
        Tguess = 3.4e-11 * (mQ/5e11)**(7/2) * (1/SC.gstar(mQ))**(1/4)
        
        TEND = 3.4e-11 * (mQ/5e11)**(7/2) * (1/SC.gstar(Tguess))**(1/4)
        
        
    if dim == 8:
        Tguess = 3.4e-20 * (mQ/5e11)**(9/2) * (1/SC.gstar(mQ))**(1/4)
        
        TEND = 3.4e-20 * (mQ/5e11)**(9/2) * (1/SC.gstar(Tguess))**(1/4)
        
    return TEND


def tend_func_lam(mQ, dim=6, lam=1.22e19):
    
    TEND=0.0
    
    mpl = 1.22e19
    
    if dim == 4: 
        Tguess = ((mQ)/(8*np.pi))**(1/2) * (1/SC.gstar(mQ))**(1/4) * ((90*mpl**2)/(8*np.pi**3))**(1/4) 
        
        TEND =  ((mQ)/(8*np.pi))**(1/2) * (1/SC.gstar(Tguess))**(1/4) * ((90*mpl**2)/(8*np.pi**3))**(1/4)  
        
    if dim == 5:
        Tguess = ((mQ**3)/(16*np.pi*lam**2))**(1/2) * (1/SC.gstar(mQ))**(1/4) * ((90*mpl**2)/(8*np.pi**3))**(1/4) 
        
        TEND =  ((mQ**3)/(16*np.pi*lam**2))**(1/2) * (1/SC.gstar(Tguess))**(1/4) * ((90*mpl**2)/(8*np.pi**3))**(1/4)  
        
    if dim == 6 : 
        Tguess = ((mQ**5)/(512*np.pi**3*lam**4))**(1/2) * (1/SC.gstar(mQ))**(1/4) * ((90*mpl**2)/(8*np.pi**3))**(1/4) 
        
        TEND =  ((mQ**5)/(512*np.pi**3*lam**4))**(1/2) * (1/SC.gstar(Tguess))**(1/4) * ((90*mpl**2)/(8*np.pi**3))**(1/4)
        
    if dim == 7:
        Tguess = ((mQ**7)/(49152*np.pi**5*lam**6))**(1/2) * (1/SC.gstar(mQ))**(1/4) * ((90*mpl**2)/(8*np.pi**3))**(1/4) 
        
        TEND =  ((mQ**7)/(49152*np.pi**5*lam**6))**(1/2) * (1/SC.gstar(Tguess))**(1/4) * ((90*mpl**2)/(8*np.pi**3))**(1/4)
        
        
    if dim == 8:
        Tguess = ((mQ**9)/(576*(4*np.pi)**7*lam**8))**(1/2) * (1/SC.gstar(mQ))**(1/4) * ((90*mpl**2)/(8*np.pi**3))**(1/4) 
        
        TEND =  ((mQ**9)/(576*(4*np.pi)**7*lam**8))**(1/2) * (1/SC.gstar(Tguess))**(1/4) * ((90*mpl**2)/(8*np.pi**3))**(1/4)
        
    if dim == 9:
        Tguess = ((mQ**11)/(11520*(4*np.pi)**9*lam**10))**(1/2) * (1/SC.gstar(mQ))**(1/4) * ((90*mpl**2)/(8*np.pi**3))**(1/4)

        TEND =  ((mQ**11)/(11520*(4*np.pi)**9*lam**10))**(1/2) * (1/SC.gstar(Tguess))**(1/4) * ((90*mpl**2)/(8*np.pi**3))**(1/4)
        
        
    return TEND


def Gamma_func_lam(mQ, dim=6, lam=1.22e19):
    
    Gamma=0.0
    
    mpl = 1.22e19
    
    if dim == 4: 
        
        Gamma = ((mQ)/(8*np.pi))
        
    if dim == 5:
        Gamma = ((mQ**3)/(16*np.pi*lam**2))
        
        
    if dim == 6 : 
        
        Gamma =  ((mQ**5)/(512*np.pi**3*lam**4))
        
    if dim == 7:
        
        Gamma =  ((mQ**7)/(49152*np.pi**5*lam**6))
        
        
    if dim == 8:
        
        Gamma  =  ((mQ**9)/(576*(4*np.pi)**7*lam**8))
        
    if dim == 9: 
        Gamma = ((mQ**11)/(11520*(4*np.pi)**9*lam**10))
        
        
    return  Gamma 

def tau_func(mQ, dim=6, lam=1.22e19):
    
    return (1/Gamma_func_lam(mQ, dim=dim, lam=lam))* 6.58e-25



if __name__ == "__main__":


    # get mQ value
    mQ = float(argv[1])
    lam = float(argv[2]) 
    
    dimension = float(argv[3])
    

    
    TEND= tend_func_lam(mQ, dim=dimension, lam=lam)
    GAMMA = Gamma_func_lam(mQ, dim=dimension, lam=lam)
    
    print("\n Decay width is %.5e GeV with Tend = %.5e \n\n" % (GAMMA, TEND))
