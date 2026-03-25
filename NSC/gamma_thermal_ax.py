import numpy as np

from scipy.interpolate import interp1d
from scipy.special import zeta
from scipy.special import kn

import scipy.integrate as integrate
import os

base_dir = os.path.dirname(os.path.realpath(__file__))

### functions for gamma_a ########
TF3, F3 = np.genfromtxt(base_dir + "/Data_gamma/F3_correction_func.csv", delimiter=',', unpack=True)

F3_interp1d = interp1d(np.log10(TF3), F3, kind='linear',  fill_value='extrapolate')
Tg3, g3_r = np.load(base_dir + "/Data_gamma/g3_run.npy")

g3_runfunc = interp1d(np.log10(Tg3), g3_r, kind='linear',  fill_value='extrapolate')

def gluon_coeff(T, mQ):
    tau = T**2 / (4*mQ**2)
    
    if tau <= 1 :
        return (1/tau)*(np.arcsin(np.sqrt(tau)))**2
    
    else: 
        return (1/tau)*(-1/4)*abs((np.log((1+np.sqrt(1-tau**(-1)))/(1-np.sqrt(1-tau**(-1)))) - 1.0j*np.pi)**2)
    
def gamma_gg(T, mQ):
    ''' normalized by fa^2 ''' 
    alpha1 = g3_runfunc(np.log10(T))
    prefact = 2*zeta(3)*8/np.pi**3
    brack1 = gluon_coeff(T, mQ) * alpha1 / (8*np.pi)
    
    return prefact * brack1**2 * F3_interp1d(np.log10(T))*T**6



def sigma_QQ_ga(s, mQ, fa):
    
    result = 0.0
    
    gs = np.sqrt(0.1179 * 4*np.pi)
    
    if np.sqrt(s) > 2*mQ:
        prefact = gs**2 / (9*np.pi*fa**2)

        kinfact = (mQ**2/s)/(1-4*(mQ**2/s))

        tanhfac = np.arctanh(np.sqrt(1-4*(mQ**2/s)))
        
        result += prefact*kinfact*tanhfac
        
    else:
        result += 0.0
    
    return result 

def sigma_Qg_Qa(s, mQ, fa):
    
    result = 0.0
    
    gs = np.sqrt(0.1179 * 4*np.pi)
    
    if np.sqrt(s) > mQ:
        prefact = gs**2 / (192*np.pi*fa**2)

        kinfact = (mQ**2/s)/(1-1.0*(mQ**2/s))

        brackfact = 4*(mQ**2/s) -  (mQ**4/s**2) - 3 - 2*np.log((mQ**2/s)) 
        
        result += prefact*kinfact*brackfact
        
    else:
        result += 0.0
    
    return result    

def kallen_func(s, m1, m2):
    return (s - (m1 + m2)**2)*(s - (m1 - m2)**2)


def gammaQQ_integrand(beta, T, mQ, fa):
    s= 4*mQ**2/(1-beta**2)
    prefact = 8*mQ**2*(beta/(1-beta**2)**2)
    return prefact*(kallen_func(s, mQ, mQ)/(np.sqrt(s))) * sigma_QQ_ga(s, mQ, fa) * kn(1, np.sqrt(s)/T) 
    #return (kallen_func(s, mQ, mQ)/(np.sqrt(s))) * sigma_QQ_ga(s, mQ, fa) * kn(1, np.sqrt(s)/T) 

    
def gammaQg_integrand(beta, T, mQ, fa):
    s= mQ**2/(1-beta**2)
    prefact = 2*mQ**2*(beta/(1-beta**2)**2)
    return prefact*(kallen_func(s, mQ, 0)/(np.sqrt(s))) * sigma_Qg_Qa(s, mQ, fa) * kn(1, np.sqrt(s)/T) 

    
def calc_gammafa2_total(T, mQ):
    fa_ref = 1.0 
    int_QQ = integrate.quad(lambda x: gammaQQ_integrand(x, T, mQ, 1.0), 0, 1.0)[0]
    int_Qg = integrate.quad(lambda x: gammaQg_integrand(x, T, mQ, 1.0), 0, 1.0)[0]
    
    gQ = 2
    gg = 2
    
    return 9*(gQ**2 * T)/(32*np.pi**4) * int_QQ  + 3*8*(gQ*gg * T)/(32*np.pi**4) * int_Qg



def setup_gamma_Qscat_funcs(mQ):
    Tspace_tmp = np.geomspace(0.01*mQ, 1e2*mQ, 300)
    MQscat_tmp = np.vectorize(calc_gammafa2_total)(Tspace_tmp, mQ) 
    
    coeff_highT = np.polyfit((np.log10(Tspace_tmp[(Tspace_tmp > (1e1*mQ)) & (Tspace_tmp < 1e2 *mQ)])),
                               (np.log10(MQscat_tmp[(Tspace_tmp > (1e1*mQ)) & (Tspace_tmp < 1e2 *mQ)])), 1)


    poly_highT = np.poly1d(coeff_highT)
    
    mQ_scats_func = interp1d(Tspace_tmp, MQscat_tmp, kind='cubic', bounds_error=False, fill_value= [0.0])
    
    return poly_highT, mQ_scats_func

def dynamic_gamma_fun(mQ, poly_highT, mQ_scats_func, T):
    
    
    if T >= 1e2*mQ:
        return 10**poly_highT(np.log10(T))
    
    if T < 1e2*mQ :
        #print(mQ_scats_func(T), gluon_coeff(T, mQ)**2*10**poly_belowmQ(np.log10(T)))
        return mQ_scats_func(T) + gamma_gg(T, mQ)

#######################################################