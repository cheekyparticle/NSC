import numpy as np

#import Qdecay_widths as QW

from . import StandardCosmo as SC 


import sys

import math

import matplotlib.pyplot as plt

import warnings

import scipy.optimize as op


from scipy.integrate import solve_ivp

from scipy.interpolate import CubicSpline 
from scipy.special import kn
from scipy.special import zeta
from scipy.interpolate import interp1d
import scipy.integrate as integrate

import scipy.optimize as op
from scipy.optimize import root

from scipy.optimize import fsolve
from . import Qdecay_widths as QW



def rho_eq_rel(g, T):
    return ((np.pi**2)/30) * g * T**4


def Yeq_nr(mQ, T, g):
    return ((45.0*g)/(2*np.pi**4*gstar_Q(T, mQ, g))) * np.sqrt((np.pi)/(8))*(mQ/T)**(3/2)* np.exp(-mQ/T)

def gstarS_Q(T, mQ, gQ):
    Teq = 10*mQ
    if T >= 1e-1 * mQ:
        return np.vectorize(SC.gstarS)(T) + (7/8)*gQ*0.5*(mQ/T)**2*kn(2, mQ/T)
    if T< 1e-1*mQ:
        return np.vectorize(SC.gstarS)(T)
    

def gstar_Q(T, mQ, gQ):
    Teq = 10*mQ
    if T >= 1e-1 * mQ: 
        return np.vectorize(SC.gstar)(T) + (7/8)*gQ*0.5*(mQ/T)**2*kn(2, mQ/T)
        
    if T< 1e-1*mQ:
        return  np.vectorize(SC.gstar)(T) 


def neq_gen_full(gi, mi, Temp, bfac=-1):
    term1 = mi**2 * Temp*kn(2, mi/Temp)
    term2 = (bfac)**1 *mi**2 * Temp*kn(2, 2*mi/Temp)/2
    term3= (bfac)**2 *mi**2 * Temp*kn(2, 3*mi/Temp)/3
    term4= (bfac)**3 *mi**2 * Temp*kn(2, 4*mi/Temp)/4
    term5= (bfac)**4 *mi**2 * Temp*kn(2, 5*mi/Temp)/5
    return gi/(2*np.pi**2)*(term1+term2 + term3+term4+term5)



def Yeq_rel(g, Temp):
    return (45/(2*np.pi**4)) * (g/ np.vectorize(SC.gstarS)(Temp))

def ni_eq_rel(gi, T):
    return (zeta(3)/np.pi**2)*gi*T**3 



def sigv(mQ):
    
    cf = 2/9 
    cg = 220/27
    nf = 3 
    alphas = 0.118
    
    return np.pi*alphas**2 * (cf * nf + cg)/(16*mQ**2)



def eqs_FO_Y(x, y, params):
    ''' x : logx, Ychi: [Yield chi], Ya : [Yield ax] '''
    
    Ychi = y[0]

    
    
    mQ = params[0]
    gQ = 12
    Temp = mQ/np.exp(x)

    
    
        
    rhoRAD = (np.pi**2/30)*gstar_Q(Temp, mQ, gQ)*Temp**4
    
    sRAD = ((2*np.pi**2)/(45))*gstarS_Q(Temp, mQ, gQ)* Temp**3
    
    
    
    Ychi_eq = neq_gen_full(gQ, mQ, Temp)/sRAD
    
    
    
    #rhoa = ga/30 * ((np.pi**(7/2))/(zeta(3)))**(4/3)*(na/ga)**(4/3)
        
    H =  np.sqrt((8/3) * np.pi * SC.GCF * (rhoRAD ))

    deltah = 1 + (1/3)*(Temp/gstarS_Q(Temp, mQ, gQ))*SC.dgstarSdT(Temp)
    sgv = sigv(mQ)
    
    
    dYchi = deltah*((sgv*sRAD)/(H)) *(Ychi_eq**2 - Ychi * Ychi) 
    
    
    
    return [dYchi]


def eqs_after_FO(x, y, params):
    ''' x : u, y: [log(fQ), log(fR)] '''
    lfQ = y[0]
    lfr = y[1]
    
    gQ = 12
    
    u = x
    
    Tin, rQ, mQ, GAMMA = params
    
    
    rhoRADi = (np.pi**2/30)*np.vectorize(SC.gstar)(Tin)*Tin**4
    
    rhoQi = rQ * rhoRADi
        
    Temp = Tin*np.exp(lfr)*np.exp(-x)

    
    rhoRAD = (np.pi**2/30)*np.vectorize(SC.gstar)(Temp)*Temp**4
    
    rhoa = (np.pi**2/30)*Temp**4
    
    sRAD = ((2*np.pi**2)/(45))*np.vectorize(SC.gstarS)(Temp)* Temp**3
    
    
    rhoQ= rhoQi * np.exp(lfQ) * np.exp(-3*u)
        

    



        
    H =  np.sqrt((8/3) * np.pi * SC.GCF * (rhoRAD + rhoQ  ))

    deltah = 1 + (1/3)*(Temp/np.vectorize(SC.gstarS)(Temp))*SC.dgstarSdT(Temp)
        
    
    #print(BRSM, BRa, Temp)

    dlfQ = - GAMMA/H 
    dlfr = 1 - 1/deltah + ( ( GAMMA) / ( 3 * H * Temp * sRAD * deltah ) ) * rhoQ 
    
    
    return [dlfQ, dlfr]

def eqs_after_decay(x, y, params):
    ''' x : u, y: [ log(fR)] '''
    lfr = y[0]
    
    gQ = 12
    
    u = x
    
    Tin = params[0]
    
    
    rhoRADi = (np.pi**2/30)*np.vectorize(SC.gstar)(Tin)*Tin**4
    
        
    Temp = Tin*np.exp(lfr)*np.exp(-x)

    
    rhoRAD = (np.pi**2/30)*np.vectorize(SC.gstar)(Temp)*Temp**4
    
    
    sRAD = ((2*np.pi**2)/(45))*np.vectorize(SC.gstarS)(Temp)* Temp**3
            

    



        
    H =  np.sqrt((8/3) * np.pi * SC.GCF * (rhoRAD ))

    deltah = 1 + (1/3)*(Temp/np.vectorize(SC.gstarS)(Temp))*SC.dgstarSdT(Temp)
        
    
    #print(BRSM, BRa, Temp)
    dlfr = 1 - 1/deltah  
    
    
    return [dlfr]


def Teq(mQ_p ,  ratio_Ti=1.0): 
    
    Ti_bFO= mQ_p*ratio_Ti
    sRADi_bFO = ((2*np.pi**2)/(45))*gstarS_Q(Ti_bFO, mQ_p, 12)* Ti_bFO**3
    rhoRADi_bFO = (np.pi**2/30)*gstar_Q(Ti_bFO, mQ_p, 12)*Ti_bFO**4
    
    Tfin_bFO = mQ_p/100 # 10 Tsig 

    params_input_FO = [mQ_p]
    
    sol_p1_bFO = solve_ivp(lambda x, y : eqs_FO_Y(x, y,
                                                  params_input_FO ),
                           [np.log(mQ_p/Ti_bFO), np.log(mQ_p/Tfin_bFO)],
                           y0=[neq_gen_full(12.0, mQ_p,Ti_bFO)/sRADi_bFO,], 
                           dense_output=True, atol=1e-10, rtol=1e-10, method='BDF')
    
    Temp_sol_p1_bFO = mQ_p/np.exp(sol_p1_bFO.t)

    sRad_sol_p1_bFO = ((2*np.pi**2)/(45))*np.vectorize(gstarS_Q)(Temp_sol_p1_bFO, mQ_p, 12)* Temp_sol_p1_bFO**3

    nQ_bFO = sRad_sol_p1_bFO*sol_p1_bFO.y[0]
    
    rhoQ_FO = nQ_bFO[-1] * mQ_p
    TFO = Temp_sol_p1_bFO[-1]
    
    Teq = (30*rhoQ_FO)/(np.pi**2 * SC.gstar(TFO)) * (1/TFO**3)
    
    return Teq 


def calc_YFO(mQ_p ,  ratio_Ti=1.0): 
    
    Ti_bFO= mQ_p*ratio_Ti
    sRADi_bFO = ((2*np.pi**2)/(45))*gstarS_Q(Ti_bFO, mQ_p, 12)* Ti_bFO**3
    rhoRADi_bFO = (np.pi**2/30)*gstar_Q(Ti_bFO, mQ_p, 12)*Ti_bFO**4
    
    Tfin_bFO = mQ_p/100 # 10 Tsig 

    params_input_FO = [mQ_p]
    
    sol_p1_bFO = solve_ivp(lambda x, y : eqs_FO_Y(x, y,
                                                  params_input_FO ),
                           [np.log(mQ_p/Ti_bFO), np.log(mQ_p/Tfin_bFO)],
                           y0=[neq_gen_full(12.0, mQ_p,Ti_bFO)/sRADi_bFO,], 
                           dense_output=True, atol=1e-10, rtol=1e-10, method='BDF')
    
    Temp_sol_p1_bFO = mQ_p/np.exp(sol_p1_bFO.t)

    sRad_sol_p1_bFO = ((2*np.pi**2)/(45))*np.vectorize(gstarS_Q)(Temp_sol_p1_bFO, mQ_p, 12)* Temp_sol_p1_bFO**3

    yQ_bFO = sol_p1_bFO.y[0]

    
    return yQ_bFO[-1]


def calc_GW_approx_input(mQ_p ,  ratio_Ti=1.0): 
    
    Ti_bFO= mQ_p*ratio_Ti
    sRADi_bFO = ((2*np.pi**2)/(45))*gstarS_Q(Ti_bFO, mQ_p, 12)* Ti_bFO**3
    rhoRADi_bFO = (np.pi**2/30)*gstar_Q(Ti_bFO, mQ_p, 12)*Ti_bFO**4
    
    Tfin_bFO = mQ_p/100 # 10 Tsig 

    params_input_FO = [mQ_p]
    
    sol_p1_bFO = solve_ivp(lambda x, y : eqs_FO_Y(x, y,
                                                  params_input_FO ),
                           [np.log(mQ_p/Ti_bFO), np.log(mQ_p/Tfin_bFO)],
                           y0=[neq_gen_full(12.0, mQ_p,Ti_bFO)/sRADi_bFO,], 
                           dense_output=True, atol=1e-10, rtol=1e-10, method='BDF')
    
    Temp_sol_p1_bFO = mQ_p/np.exp(sol_p1_bFO.t)

    sRad_sol_p1_bFO = ((2*np.pi**2)/(45))*np.vectorize(gstarS_Q)(Temp_sol_p1_bFO, mQ_p, 12)* Temp_sol_p1_bFO**3

    yQ_bFO = sol_p1_bFO.y[0]
    
    nQ_bFO = sRad_sol_p1_bFO*sol_p1_bFO.y[0]


    rhoQ_FO = nQ_bFO[-1] * mQ_p
    TFO = Temp_sol_p1_bFO[-1]
    
    Teq = (30*rhoQ_FO)/(np.pi**2 * SC.gstar(TFO)) * (1/TFO**3)
    
    return yQ_bFO[-1], Teq


def afin(aexp, rQi, rRadi, t, ail):

    a = [10.**(aexp[0])]

    ain = 10.**ail # Initial scale factor
    
    A = -ain * rQi * np.sqrt(SC.GCF * (ain * rQi + rRadi))
    B = a[0] * rQi * np.sqrt(SC.GCF * (a[0] * rQi + rRadi))
    C = 2. * rRadi * (np.sqrt(SC.GCF*(ain * rQi + rRadi)) - np.sqrt(SC.GCF*(a[0]*rQi + rRadi)))
    D = SC.GCF * np.sqrt(6.*np.pi) * rQi**2

    
    #print(A + B + C - D*t)
    
    return [A + B + C - D*t]


def Track_all_lam(mQ_p , d_decay = 6, lam=1.22e19, ratio_Ti=1.0): 
    
    Ti_bFO= mQ_p*ratio_Ti
    sRADi_bFO = ((2*np.pi**2)/(45))*gstarS_Q(Ti_bFO, mQ_p, 12)* Ti_bFO**3
    rhoRADi_bFO = (np.pi**2/30)*gstar_Q(Ti_bFO, mQ_p, 12)*Ti_bFO**4
    
    Tfin_bFO = mQ_p/100 # 10 Tsig 

    params_input_FO = [mQ_p]
    
    sol_p1_bFO = solve_ivp(lambda x, y : eqs_FO_Y(x, y,
                                                  params_input_FO ),
                           [np.log(mQ_p/Ti_bFO), np.log(mQ_p/Tfin_bFO)],
                           y0=[neq_gen_full(12.0, mQ_p,Ti_bFO)/sRADi_bFO,], 
                           dense_output=True, atol=1e-10, rtol=1e-10, method='BDF')
    
    Temp_sol_p1_bFO = mQ_p/np.exp(sol_p1_bFO.t)

    sRad_sol_p1_bFO = ((2*np.pi**2)/(45))*np.vectorize(gstarS_Q)(Temp_sol_p1_bFO, mQ_p, 12)* Temp_sol_p1_bFO**3

    rhor_bFO = np.pi**2/(30)*np.vectorize(gstar_Q)(Temp_sol_p1_bFO, mQ_p, 12) * Temp_sol_p1_bFO**4

    nQ_bFO = sRad_sol_p1_bFO*sol_p1_bFO.y[0]
    
    rhoQ_FO = nQ_bFO[-1] * mQ_p
    TFO = Temp_sol_p1_bFO[-1]

    ######## solve for the decay ############

    Ti_aFO = TFO

    rhoSMi_aFO = (np.pi**2/30)*np.vectorize(SC.gstar)(Ti_aFO)*Ti_aFO**4

    rhoQf_bFO = mQ_p*nQ_bFO[-1] + 3/2 * nQ_bFO[-1] * Temp_sol_p1_bFO[-1]

    rhoQ_bFO = mQ_p*nQ_bFO + 3/2 * nQ_bFO * Temp_sol_p1_bFO
    
    ratioQ_aFO= rhoQf_bFO/rhoSMi_aFO

    gamma = QW.Gamma_func_lam(mQ_p, dim=d_decay, lam=lam)
    TEND = QW.tend_func_lam(mQ_p, dim=d_decay, lam=lam)

    if Ti_aFO < TEND:

        print("here, %.3e, %.3e\n" % (Ti_aFO, TEND))
        Rin_aFO = 1.0
        Tfin_aFO = 1e-3 # 10 Tsig 
        Rfin_aFO = (np.vectorize(SC.gstarS)(Ti_aFO)/np.vectorize(SC.gstarS)(Tfin_aFO))**(1/3) * ((Ti_aFO*Rin_aFO)/(Tfin_aFO))

        uin_aFO, ufin_aFO = 0.0, np.log(Rfin_aFO/Rin_aFO) 

        params_input_p1_aFO = [Ti_aFO]

        sol_p1_aFO = solve_ivp(lambda x, y : eqs_after_decay(x, y, params_input_p1_aFO ), [uin_aFO, ufin_aFO],
                            y0=[np.log(1.0)], method='BDF')

        Temp_sol_p1_aFO = Ti_aFO*np.exp(sol_p1_aFO.y[0])*np.exp(-sol_p1_aFO.t)

        rhor_aFO = (np.pi**2/30)*np.vectorize(SC.gstar)(Temp_sol_p1_aFO)*Temp_sol_p1_aFO**4
        rhoQ_aFO = np.zeros(np.size(sol_p1_aFO.t))
        sRad_sol_p1_aFO = SC.sRAD(Temp_sol_p1_aFO)

        sRad_volume_aFO = sRad_sol_p1_aFO * np.exp(3* sol_p1_aFO.t)

        rhor = np.concatenate((rhor_bFO, rhor_aFO), axis=0)

        rhoQ = np.concatenate((rhoQ_bFO, rhoQ_aFO), axis=0)

        sRAD = np.concatenate((sRad_sol_p1_bFO, sRad_sol_p1_aFO), axis=0)


        Temp = np.concatenate((Temp_sol_p1_bFO ,Temp_sol_p1_aFO), axis=0)

    else: 
        lifetimeQ = 1/gamma


        #Tfin_aFO = 1e-8 # 10 Tsig 

        Rin_aFO = 1.0
        Rfin_aFO = 10**root(afin, [20.], args = (rhoQf_bFO, rhoSMi_aFO, lifetimeQ, 0.0), method='lm', tol=1e-40).x * 1e1

        #Rfin_aFO = (np.vectorize(SC.gstarS)(Ti_aFO)/np.vectorize(SC.gstarS)(Tfin_aFO))**(1/3) * ((Ti_aFO*Rin_aFO)/(Tfin_aFO))

        uin_aFO, ufin_aFO = 0.0, np.log(Rfin_aFO/Rin_aFO) 

        params_input_p1_aFO = [Ti_aFO, ratioQ_aFO, mQ_p, gamma]

        sol_p1_aFO = solve_ivp(lambda x, y : eqs_after_FO(x, y, params_input_p1_aFO ), [uin_aFO, ufin_aFO],
                            y0=[np.log(1.0), np.log(1.0)], atol=1e-14, rtol=1e-10, 
                            dense_output=True, method='BDF')

        Temp_sol_p1_aFO = Ti_aFO*np.exp(sol_p1_aFO.y[1])*np.exp(-sol_p1_aFO.t)

        rhor_aFO = (np.pi**2/30)*np.vectorize(SC.gstar)(Temp_sol_p1_aFO)*Temp_sol_p1_aFO**4
        rhoQ_aFO = ratioQ_aFO * rhor_aFO[1] * np.exp(sol_p1_aFO.y[0]) * np.exp(-3*sol_p1_aFO.t)

        sRad_sol_p1_aFO = ((2*np.pi**2)/(45))*np.vectorize(gstarS_Q)(Temp_sol_p1_aFO, mQ_p, 12)* Temp_sol_p1_aFO**3

        sRad_volume_aFO = sRad_sol_p1_aFO * np.exp(3* sol_p1_aFO.t)

        if Temp_sol_p1_aFO[-1] < 1e-3: 

            print("final temp %.3e\n" % (Temp_sol_p1_aFO[-1]))
            rhor = np.concatenate((rhor_bFO, rhor_aFO), axis=0)

            rhoQ = np.concatenate((rhoQ_bFO, rhoQ_aFO), axis=0)

            sRAD = np.concatenate((sRad_sol_p1_bFO, sRad_sol_p1_aFO), axis=0)


            Temp = np.concatenate((Temp_sol_p1_bFO ,Temp_sol_p1_aFO), axis=0)

        else: 
            print("final temp %.3e\n" % (Temp_sol_p1_aFO[-1]))
            Ti_adec = Temp_sol_p1_aFO[-1]
            Rin_adec = 1.0
            Tfin_adec = 1e-3 # 10 Tsig 
            Rfin_adec = (np.vectorize(SC.gstarS)(Ti_adec)/np.vectorize(SC.gstarS)(Tfin_adec))**(1/3) * ((Ti_adec*Rin_adec)/(Tfin_adec))

            uin_aFO, ufin_aFO = 0.0, np.log(Rfin_aFO/Rin_aFO) 

            params_input_p1_adec = [Ti_adec]

            sol_p1_adec = solve_ivp(lambda x, y : eqs_after_decay(x, y, params_input_p1_adec ), [uin_aFO, ufin_aFO],
                                y0=[np.log(1.0)], method='BDF')

            Temp_sol_p1_adec = Ti_adec*np.exp(sol_p1_adec.y[0])*np.exp(-sol_p1_adec.t)

            rhor_adec = (np.pi**2/30)*np.vectorize(SC.gstar)(Temp_sol_p1_adec)*Temp_sol_p1_adec**4
            rhoQ_adec = np.zeros(np.size(sol_p1_adec.t))
            sRad_sol_p1_adec = SC.sRAD(Temp_sol_p1_adec)

            rhor = np.concatenate((rhor_bFO, rhor_aFO, rhor_adec), axis=0)

            rhoQ = np.concatenate((rhoQ_bFO, rhoQ_aFO, rhoQ_adec), axis=0)

            sRAD = np.concatenate((sRad_sol_p1_bFO, sRad_sol_p1_aFO, sRad_sol_p1_adec), axis=0)

            Temp = np.concatenate((Temp_sol_p1_bFO ,Temp_sol_p1_aFO, Temp_sol_p1_adec), axis=0)
        
    return rhor, rhoQ, sRAD, Temp, sol_p1_aFO.t, sRad_volume_aFO, rhoQ_aFO, rhor_aFO, ufin_aFO


def GW_input_lam(mQ_p , d_decay = 6, lam=1.22e19, ratio_Ti=1.0): 
    '''return sRad_volume_aFO[-1]/sRad_volume_aFO[0] , Teq, Tdec'''
    
    Ti_bFO= mQ_p*ratio_Ti
    sRADi_bFO = ((2*np.pi**2)/(45))*gstarS_Q(Ti_bFO, mQ_p, 12)* Ti_bFO**3
    rhoRADi_bFO = (np.pi**2/30)*gstar_Q(Ti_bFO, mQ_p, 12)*Ti_bFO**4
    
    Tfin_bFO = mQ_p/100 # 10 Tsig 

    params_input_FO = [mQ_p]
    
    sol_p1_bFO = solve_ivp(lambda x, y : eqs_FO_Y(x, y,
                                                  params_input_FO ),
                           [np.log(mQ_p/Ti_bFO), np.log(mQ_p/Tfin_bFO)],
                           y0=[neq_gen_full(12.0, mQ_p,Ti_bFO)/sRADi_bFO,], 
                           dense_output=True, atol=1e-10, rtol=1e-10, method='BDF')
    
    Temp_sol_p1_bFO = mQ_p/np.exp(sol_p1_bFO.t)

    sRad_sol_p1_bFO = ((2*np.pi**2)/(45))*np.vectorize(gstarS_Q)(Temp_sol_p1_bFO, mQ_p, 12)* Temp_sol_p1_bFO**3

    rhor_bFO = np.pi**2/(30)*np.vectorize(gstar_Q)(Temp_sol_p1_bFO, mQ_p, 12) * Temp_sol_p1_bFO**4

    nQ_bFO = sRad_sol_p1_bFO*sol_p1_bFO.y[0]
    
    rhoQ_FO = nQ_bFO[-1] * mQ_p
    TFO = Temp_sol_p1_bFO[-1]

    ######## solve for the decay ############

    Ti_aFO = TFO

    rhoSMi_aFO = (np.pi**2/30)*np.vectorize(SC.gstar)(Ti_aFO)*Ti_aFO**4

    rhoQf_bFO = mQ_p*nQ_bFO[-1] + 3/2 * nQ_bFO[-1] * Temp_sol_p1_bFO[-1]

    rhoQ_bFO = mQ_p*nQ_bFO + 3/2 * nQ_bFO * Temp_sol_p1_bFO
    
    ratioQ_aFO= rhoQf_bFO/rhoSMi_aFO

    gamma = QW.Gamma_func_lam(mQ_p, dim=d_decay, lam=lam)
    TEND = QW.tend_func_lam(mQ_p, dim=d_decay, lam=lam)

    if Ti_aFO < TEND:

        #print("here, %.3e, %.3e\n" % (Ti_aFO, TEND))
        Rin_aFO = 1.0
        Tfin_aFO = 1e-3 # 10 Tsig 
        Rfin_aFO = (np.vectorize(SC.gstarS)(Ti_aFO)/np.vectorize(SC.gstarS)(Tfin_aFO))**(1/3) * ((Ti_aFO*Rin_aFO)/(Tfin_aFO))

        uin_aFO, ufin_aFO = 0.0, np.log(Rfin_aFO/Rin_aFO) 

        params_input_p1_aFO = [Ti_aFO]

        sol_p1_aFO = solve_ivp(lambda x, y : eqs_after_decay(x, y, params_input_p1_aFO ), [uin_aFO, ufin_aFO],
                            y0=[np.log(1.0)], method='BDF')

        Temp_sol_p1_aFO = Ti_aFO*np.exp(sol_p1_aFO.y[0])*np.exp(-sol_p1_aFO.t)

        rhor_aFO = (np.pi**2/30)*np.vectorize(SC.gstar)(Temp_sol_p1_aFO)*Temp_sol_p1_aFO**4
        rhoQ_aFO = np.zeros(np.size(sol_p1_aFO.t))
        sRad_sol_p1_aFO = SC.sRAD(Temp_sol_p1_aFO)

        sRad_volume_aFO = sRad_sol_p1_aFO * np.exp(3* sol_p1_aFO.t)

        rhor = np.concatenate((rhor_bFO, rhor_aFO), axis=0)

        rhoQ = np.concatenate((rhoQ_bFO, rhoQ_aFO), axis=0)

        sRAD = np.concatenate((sRad_sol_p1_bFO, sRad_sol_p1_aFO), axis=0)


        Temp = np.concatenate((Temp_sol_p1_bFO ,Temp_sol_p1_aFO), axis=0)

    else: 
        lifetimeQ = 1/gamma


        #Tfin_aFO = 1e-8 # 10 Tsig 

        Rin_aFO = 1.0
        Rfin_aFO = 10**root(afin, [20.], args = (rhoQf_bFO, rhoSMi_aFO, lifetimeQ, 0.0), method='lm', tol=1e-40).x * 1e1

        #Rfin_aFO = (np.vectorize(SC.gstarS)(Ti_aFO)/np.vectorize(SC.gstarS)(Tfin_aFO))**(1/3) * ((Ti_aFO*Rin_aFO)/(Tfin_aFO))

        uin_aFO, ufin_aFO = 0.0, np.log(Rfin_aFO/Rin_aFO) 

        params_input_p1_aFO = [Ti_aFO, ratioQ_aFO, mQ_p, gamma]

        sol_p1_aFO = solve_ivp(lambda x, y : eqs_after_FO(x, y, params_input_p1_aFO ), [uin_aFO, ufin_aFO],
                            y0=[np.log(1.0), np.log(1.0)], atol=1e-14, rtol=1e-10, 
                            dense_output=True, method='BDF')

        Temp_sol_p1_aFO = Ti_aFO*np.exp(sol_p1_aFO.y[1])*np.exp(-sol_p1_aFO.t)

        rhor_aFO = (np.pi**2/30)*np.vectorize(SC.gstar)(Temp_sol_p1_aFO)*Temp_sol_p1_aFO**4
        rhoQ_aFO = ratioQ_aFO * rhor_aFO[1] * np.exp(sol_p1_aFO.y[0]) * np.exp(-3*sol_p1_aFO.t)

        sRad_sol_p1_aFO = ((2*np.pi**2)/(45))*np.vectorize(gstarS_Q)(Temp_sol_p1_aFO, mQ_p, 12)* Temp_sol_p1_aFO**3

        sRad_volume_aFO = sRad_sol_p1_aFO * np.exp(3* sol_p1_aFO.t)

        if Temp_sol_p1_aFO[-1] < 1e-3: 

            #print("final temp %.3e\n" % (Temp_sol_p1_aFO[-1]))
            rhor = np.concatenate((rhor_bFO, rhor_aFO), axis=0)

            rhoQ = np.concatenate((rhoQ_bFO, rhoQ_aFO), axis=0)

            sRAD = np.concatenate((sRad_sol_p1_bFO, sRad_sol_p1_aFO), axis=0)


            Temp = np.concatenate((Temp_sol_p1_bFO ,Temp_sol_p1_aFO), axis=0)

        else: 
            #print("final temp %.3e\n" % (Temp_sol_p1_aFO[-1]))
            Ti_adec = Temp_sol_p1_aFO[-1]
            Rin_adec = 1.0
            Tfin_adec = 1e-3 # 10 Tsig 
            Rfin_adec = (np.vectorize(SC.gstarS)(Ti_adec)/np.vectorize(SC.gstarS)(Tfin_adec))**(1/3) * ((Ti_adec*Rin_adec)/(Tfin_adec))

            uin_aFO, ufin_aFO = 0.0, np.log(Rfin_aFO/Rin_aFO) 

            params_input_p1_adec = [Ti_adec]

            sol_p1_adec = solve_ivp(lambda x, y : eqs_after_decay(x, y, params_input_p1_adec ), [uin_aFO, ufin_aFO],
                                y0=[np.log(1.0)], method='BDF')

            Temp_sol_p1_adec = Ti_adec*np.exp(sol_p1_adec.y[0])*np.exp(-sol_p1_adec.t)

            rhor_adec = (np.pi**2/30)*np.vectorize(SC.gstar)(Temp_sol_p1_adec)*Temp_sol_p1_adec**4
            rhoQ_adec = np.zeros(np.size(sol_p1_adec.t))
            sRad_sol_p1_adec = SC.sRAD(Temp_sol_p1_adec)

            rhor = np.concatenate((rhor_bFO, rhor_aFO, rhor_adec), axis=0)

            rhoQ = np.concatenate((rhoQ_bFO, rhoQ_aFO, rhoQ_adec), axis=0)

            sRAD = np.concatenate((sRad_sol_p1_bFO, sRad_sol_p1_aFO, sRad_sol_p1_adec), axis=0)

            Temp = np.concatenate((Temp_sol_p1_bFO ,Temp_sol_p1_aFO, Temp_sol_p1_adec), axis=0)

    
    Teq = 0.0 
    Tdec = 0.0
    if any((rhoQ/rhor) > 1):
        Teq = max(Temp[(rhoQ/rhor) > 1])
        Tdec = min(Temp[(rhoQ/rhor) > 1])
        
    
    return sRad_volume_aFO[-1]/sRad_volume_aFO[0] , Teq, Tdec