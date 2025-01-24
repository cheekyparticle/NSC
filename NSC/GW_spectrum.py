from scipy.special import spherical_jn
import ACcosmo.StandardCosmo as SC
import numpy as np
import solve_Cosmo as TEQ
import Qdecay_widths as QW



def T1(x):
    T1sqr = 1 + 1.57*x + 3.42*x**2
    
    return T1sqr

def T2(x):
    T2sqr = (1 - 0.22*x**(3/2) + 0.65 * x**2)**(-1)
    return T2sqr

def T3(x): 
    T3sqr = 1 + 0.59*x + 0.65*x**2
    
    return T3sqr


import natpy as nat


def kRH(TRH):
    gs0 = SC.gstarS(2.3e-13)
    mpc2s = (nat.Mpc**(-1)).convert(nat.s**(-1))
    
    return (1.7e14 *mpc2s) * (SC.gstarS(TRH)/gs0)**(1/6)*(TRH/1e7)

def keq(omegam=0.3, h=0.7 ): 
    
    mpc2s = (nat.Mpc**(-1)).convert(nat.s**(-1))
    return (7.1e-2 *mpc2s) * omegam*h**2
    
    
def kdec(Tdec):
    gs0 = SC.gstarS(2.3e-13)
    mpc2s = (nat.Mpc**(-1)).convert(nat.s**(-1))
    return (1.7e14 * mpc2s) *(SC.gstarS(Tdec)/gs0)**(1/6) * (Tdec/1e7)
    
    
def Tosc_basic(ma):
    ''' eq 15 of https://arxiv.org/pdf/2107.13588.pdf'''
    # ma is in eV and needs to be in GeV
    
    ma=1e-9*ma
    MPL = 2.4e18
    TQCD = 0.150
    T_bQCD = ((1/np.pi)*np.sqrt((10)/SC.gstar(TQCD))*ma * MPL)**(1/2)
    T_bQCD = ((1/np.pi)*np.sqrt((10)/SC.gstar(T_bQCD))*ma * MPL)**(1/2)
    
    if T_bQCD > TQCD:
        T_aQCD = ((1/np.pi)*np.sqrt((10)/SC.gstar(TQCD))*ma * MPL*TQCD**4)**(1/6)
        T_aQCD = ((1/np.pi)*np.sqrt((10)/SC.gstar(T_aQCD))*ma * MPL*TQCD**4)**(1/6)
        
        return T_aQCD
    
    else: 
        return T_bQCD
    
    
def FormStandard(k, TRH): 
    
    return T1(k/keq()) * T2(k/kRH(TRH))



def Dilution_func(Tdecay, Teq , beta=3.0):
    return ((4/beta**2)*(Tdecay/Teq)**(4-beta))**(3/beta)

def Dilution_func_EMD(Tdecay, Teq ):
    return (SC.gstar(Tdecay)/SC.gstar(Teq)) * (SC.gstarS(Tdecay)/SC.gstarS(Teq)) * (Tdecay/Teq)**(3/4)
    #return ((4/beta**2)*(Tdecay/Teq)**(4-beta))**(3/beta)
    

def FormIMD(k, Tdec, Teq, TRH): 
    Dil = max(Dilution_func(Tdec, Teq) **(-1), 1)

    return T1(k/keq()) * T2(k/kdec(Tdec)) * T3(k/(kdec(Tdec) * Dil**(2/3))) * T2(k/(kRH(TRH)* Dil**(-1/3))) 


def FormIMD_new(k, Tdec, Teq, TRH): 
    Dil = max(Dilution_func_EMD(Tdec, Teq) **(-1), 1)

    return T1(k/keq()) * T2(k/kdec(Tdec)) * T3(k/(kdec(Tdec) * Dil**(2/3))) * T2(k/(kRH(TRH)* Dil**(-1/3))) 

def FormIMD_dil(k, Tdec, Teq, TRH, dil): 

    return T1(k/keq()) * T2(k/kdec(Tdec)) * T3(k/(kdec(Tdec) * dil**(2/3))) * T2(k/(kRH(TRH)* dil**(-1/3))) 

def Tin(k):
    mpcinv_to_Hz = 1.54e-15
    mpc2s = 1.027e14 

    Tguess = 100
    Temp1 = 5.8e6 * (106.75/SC.gstarS(Tguess))**(1/6) * (k / (1e4 / mpc2s))
    
    Temp2 = 5.8e6 * (106.75/SC.gstarS(Temp1))**(1/6) * (k / (1e4 / mpc2s))
    
    return 5.8e6 * (106.75/SC.gstarS(Temp2))**(1/6) * (k / (1e4 /mpc2s))

def T2T(k, TRH, omegam=0.31, h = 0.7):
    Temp0 = 2.3e-13
    Hub0 = h * (9.777)**(-1) * 3.168e-17
    zk = 2*k/Hub0
    
    #return omegam**2 * (SC.gstar(Tin(k))/SC.gstar(Temp0)) * (SC.gstarS(Temp0)/SC.gstarS(Tin(k)))**(4/3) * (spherical_jn(1, zk)/zk)**2 * FormStandard(k, TRH)
    #print(SC.gstarS(Tin(k)), Tin(k))
    #return omegam**2 * (SC.gstar(Tin(k))/SC.gstar(Temp0)) * (SC.gstarS(Temp0)/SC.gstarS(Tin(k)))**(4/3) * (9/(zk)**4) * FormStandard(k, TRH)
    return omegam**2 * (SC.gstar(k)/SC.gstar(Temp0)) * (SC.gstarS(Temp0)/SC.gstarS(k))**(4/3) * (9/(zk)**4) * FormStandard(k, TRH)


def T2TIMD(k,  Tdec, Teq, TRH, omegam=0.31, h = 0.7):
    Temp0 = 2.3e-13
    Hub0 = h * (9.777)**(-1) * 3.168e-17
    zk = 2*k/Hub0
    
    #return omegam**2 * (SC.gstar(Tin(k))/SC.gstar(Temp0)) * (SC.gstarS(Temp0)/SC.gstarS(Tin(k)))**(4/3) * (spherical_jn(1, zk)/zk)**2 * FormStandard(k, TRH)
    #print(SC.gstarS(Tin(k)), Tin(k))
    #return omegam**2 * (SC.gstar(Tin(k))/SC.gstar(Temp0)) * (SC.gstarS(Temp0)/SC.gstarS(Tin(k)))**(4/3) * (9/(zk)**4) * FormStandard(k, TRH)
    return omegam**2 * (SC.gstar(k)/SC.gstar(Temp0)) * (SC.gstarS(Temp0)/SC.gstarS(k))**(4/3) * (9/(zk)**4) *FormIMD(k, Tdec, Teq, TRH)


def T2TIMD_new(k,  Tdec, Teq, TRH, omegam=0.31, h = 0.7):
    Temp0 = 2.3e-13
    Hub0 = h * (9.777)**(-1) * 3.168e-17
    zk = 2*k/Hub0
    
    #return omegam**2 * (SC.gstar(Tin(k))/SC.gstar(Temp0)) * (SC.gstarS(Temp0)/SC.gstarS(Tin(k)))**(4/3) * (spherical_jn(1, zk)/zk)**2 * FormStandard(k, TRH)
    #print(SC.gstarS(Tin(k)), Tin(k))
    #return omegam**2 * (SC.gstar(Tin(k))/SC.gstar(Temp0)) * (SC.gstarS(Temp0)/SC.gstarS(Tin(k)))**(4/3) * (9/(zk)**4) * FormStandard(k, TRH)
    return omegam**2 * (SC.gstar(k)/SC.gstar(Temp0)) * (SC.gstarS(Temp0)/SC.gstarS(k))**(4/3) * (9/(zk)**4) *FormIMD_new(k, Tdec, Teq, TRH)

def T2TIMD_dil(k,  Tdec, Teq, TRH, dil, omegam=0.31, h = 0.7):
    Temp0 = 2.3e-13
    Hub0 = h * (9.777)**(-1) * 3.168e-17
    zk = 2*k/Hub0
    
    #return omegam**2 * (SC.gstar(Tin(k))/SC.gstar(Temp0)) * (SC.gstarS(Temp0)/SC.gstarS(Tin(k)))**(4/3) * (spherical_jn(1, zk)/zk)**2 * FormStandard(k, TRH)
    #print(SC.gstarS(Tin(k)), Tin(k))
    #return omegam**2 * (SC.gstar(Tin(k))/SC.gstar(Temp0)) * (SC.gstarS(Temp0)/SC.gstarS(Tin(k)))**(4/3) * (9/(zk)**4) * FormStandard(k, TRH)
    return omegam**2 * (SC.gstar(k)/SC.gstar(Temp0)) * (SC.gstarS(Temp0)/SC.gstarS(k))**(4/3) * (9/(zk)**4) *FormIMD_dil(k, Tdec, Teq, TRH, dil)

def PT_func(k, nT, r=0.035, Ask= 2.0989e-9, kstar=0.05):
    mpcinv_to_Hz = 1.54e-15
    mpc2s = 1.027e14
    kstar = kstar*(1/mpc2s)
    
    return r * Ask * (k/kstar)**(nT)


def OmegaGW(k, TRH, nT, h=0.7):
    a0 = 1 
    Hub0 = h * (9.777)**(-1) * 3.168e-17

    # return (1/12)*(k/(a0 * 2.2e-4))**2 * T2T(k, TRH) * PT_func(k,nT)
    return (1/12)*(k/(a0 * Hub0))**2 * T2T(k, TRH) * PT_func(k,nT)


def OmegaGW_IMD(k,Tdec, Teq,  TRH, nT, h=0.7):
    a0 = 1 
    Hub0 = h * (9.777)**(-1) * 3.168e-17

    # return (1/12)*(k/(a0 * 2.2e-4))**2 * T2T(k, TRH) * PT_func(k,nT)
    return (1/12)*(k/(a0 * Hub0))**2 * T2TIMD(k, Tdec, Teq, TRH) * PT_func(k,nT)

def OmegaGW_IMD_new(k,Tdec, Teq,  TRH, nT, h=0.7):
    a0 = 1 
    Hub0 = h * (9.777)**(-1) * 3.168e-17

    # return (1/12)*(k/(a0 * 2.2e-4))**2 * T2T(k, TRH) * PT_func(k,nT)
    return (1/12)*(k/(a0 * Hub0))**2 * T2TIMD_new(k, Tdec, Teq, TRH) * PT_func(k,nT)

def OmegaGW_preferred_axion_constant(mQ_p, d_decay=6):
    """
    Precomputes the constants that do not depend on k.
    """
    a0 = 1
    h = 0.7
    Hub0 = h * (9.777)**(-1) * 3.168e-17

    # Precompute values from TEQ.GW_input
    invdil, Teq, Tdec = TEQ.GW_input(mQ_p, d_decay=d_decay)

    return Hub0, invdil, Teq, Tdec, a0


def OmegaGW_preferred_axion(k_values, mQ_p, TRH, nT, d_decay=6, h=0.7):
    """
    Computes OmegaGW for multiple k values, using precomputed constants
    to avoid redundant calculations.
    """
    # Precompute constants
    Hub0, invdil, Teq, Tdec, a0 = OmegaGW_preferred_axion_constant(mQ_p, d_decay)

    results = []
    if Teq == 0:
        # If Teq == 0, return the result of OmegaGW for all k values
        for k in k_values:
            results.append(OmegaGW(k, TRH, nT, h=h))
    else:
        # Compute OmegaGW for each k
        for k in k_values:
            result = (
                (1/12) * (k / (a0 * Hub0))**2
                * T2TIMD_dil(k, Tdec, Teq, TRH, invdil)
                * PT_func(k, nT)
            )
            results.append(result)

    return results

def OmegaGW_preferred_axion_constant_lam(mQ_p, d_decay=6, lam=1.22e19):
    """
    Precomputes the constants that do not depend on k.
    """
    a0 = 1
    h = 0.7
    Hub0 = h * (9.777)**(-1) * 3.168e-17

    # Precompute values from TEQ.GW_input
    invdil, Teq, Tdec = TEQ.GW_input_lam(mQ_p, d_decay=d_decay, lam=lam)

    return Hub0, invdil, Teq, Tdec, a0


def OmegaGW_preferred_axion_lam(k_values, mQ_p, TRH, nT, d_decay=6, lam=1.22e19, h=0.7):
    """
    Computes OmegaGW for multiple k values, using precomputed constants
    to avoid redundant calculations.
    """
    # Precompute constants
    Hub0, invdil, Teq, Tdec, a0 = OmegaGW_preferred_axion_constant_lam(mQ_p, d_decay, lam)
    #print(invdil, Teq, Tdec)
    results = []
    if Teq == 0:
        # If Teq == 0, return the result of OmegaGW for all k values
        for k in k_values:
            results.append(OmegaGW(k, TRH, nT, h=h))
    else:
        # Compute OmegaGW for each k
        for k in k_values:
            result = (
                (1/12) * (k / (a0 * Hub0))**2
                * T2TIMD_dil(k, Tdec, Teq, TRH, invdil)
                * PT_func(k, nT)
            )
            results.append(result)

    return results


def entropy_turner(mQ, Y, dim):

    a =  2 * np.pi**2 *SC.gstar(QW.tend_func_lam(mQ, dim=dim))/45 
    GAM = QW.Gamma_func_lam(mQ, dim=dim)

    
    return (1  + 2.95 * a**(1/3) * (Y*mQ)**(4/3)/(1.22e19 * GAM )**(2/3))**(-3/4)



def OmegaGW_preferred_axion_constant_approx(mQ_p, d_decay=6, lam= 1.22e19):
    """
    Precomputes the constants that do not depend on k.
    """
    a0 = 1
    h = 0.7
    Hub0 = h * (9.777)**(-1) * 3.168e-17

    # Precompute values from TEQ.GW_input
    Yield, Teq = TEQ.calc_GW_approx_input(mQ_p)

    dil = entropy_turner(mQ_p, Yield, d_decay)**(-1)

    Tdec = QW.tend_func_lam(mQ_p, dim=d_decay, lam=lam)

    return Hub0, dil, Teq, Tdec, a0


def OmegaGW_preferred_axion_approx(k_values, mQ_p, TRH, nT, d_decay=6, lam=1.22e19, h=0.7):
    """
    Computes OmegaGW for multiple k values, using precomputed constants
    to avoid redundant calculations.
    """
    # Precompute constants
    Hub0, invdil, Teq, Tdec, a0 = OmegaGW_preferred_axion_constant_approx(mQ_p, d_decay, lam)
    print(invdil, Teq, Tdec)
    results = []
    if (Teq == 0 ) or (Teq < Tdec):
        # If Teq == 0, return the result of OmegaGW for all k values
        for k in k_values:
            results.append(OmegaGW(k, TRH, nT, h=h))
    else:
        # Compute OmegaGW for each k
        for k in k_values:
            result = (
                (1/12) * (k / (a0 * Hub0))**2
                * T2TIMD_dil(k, Tdec, Teq, TRH, invdil)
                * PT_func(k, nT)
            )
            results.append(result)

    return results



LIGOf, LIGO_OmGW = np.genfromtxt('../gwdetectorsdatafile-latest/LIGO.dat', unpack = True)

LISAf, LISA_OmGW = np.genfromtxt('../gwdetectorsdatafile-latest/lisa.dat', unpack = True)


Nanof, Nano_OmGW = np.genfromtxt('../gwdetectorsdatafile-latest/Nano22.dat', unpack = True)
Nanobf, Nanob_OmGW = np.genfromtxt('../gwdetectorsdatafile-latest/Nano22-b.dat', unpack = True)

SKA20f ,  SKA20_om = np.genfromtxt('../gwdetectorsdatafile-latest/SKA2.dat', unpack = True)

ETf ,  ET_om = np.genfromtxt('../gwdetectorsdatafile-latest/ET.dat', unpack = True)