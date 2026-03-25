
import numpy as np

from NSC.solve_Cosmo import neq_gen_full, sigv, ni_eq_rel, Yeq_rel, afin, gstar_Q, gstarS_Q
from NSC import gamma_thermal_ax as gam


from . import StandardCosmo as SC 

from scipy.special import kn

from scipy.special import zeta

from scipy.optimize import fsolve, root
from scipy.integrate import solve_ivp

import warnings




############################# Model dependen things ##############################

def sigv(mQ):
    
    cf = 2/9 
    cg = 220/27
    nf = 3 
    alphas = 0.118
    
    return np.pi*alphas**2 * (cf * nf + cg)/(16*mQ**2)

def gamma_dW3( mQ, fa, lambda_3d=1.0, md=4.18, Lambda=1.2e19):
    gEW = 0.64
    term1 = (gEW**2) / (256* np.pi)
    term2 = ((lambda_3d * md * fa**2) / (Lambda * mQ**2))**2
    return term1 * term2 * mQ

def gamma_uW( mQ, fa, lambda_3d=1.0, md=4.18, Lambda=1.2e19):
    gEW = 0.64
    term1 = (gEW**2) / (128* np.pi)
    term2 = ((lambda_3d * md * fa**2) / (Lambda * mQ**2))**2
    return term1 * term2 * mQ

def gamma_dB( mQ, fa, lambda_3d=1.0, md=4.18, Lambda=1.2e19):
    gEW_prime = 0.34
    term1 = (gEW_prime**2) / (256 * np.pi)
    term2 = ((lambda_3d * md * fa**2) / (Lambda * mQ**2))**2
    return term1 * term2 * mQ

def gamma_ad( mQ, fa, lambda_3d=1.0, Lambda=1.2e19):
    term = (1 / (32 * np.pi)) * ((lambda_3d**2 * fa**2) / (Lambda**2))
    return 3*term * mQ 

def gamma_aad( mQ, fa, lambda_3d=1.0, Lambda=1.2e19):
    term = (lambda_3d**2 / Lambda**2) * (mQ**3 / (768 * np.pi**3))
    return 3*term 

################## differential equations to solve ###########

def eqs_eqdecay_Y(x, y, params):
    ''' x : logx, y: [Yield] '''
    
    ''' x : logx, Ychi: [Yield chi], Ya : [Yield ax] '''

    Ya = y[0]
    
    
    mQ, GAMMA, fa, poly_highT, mQ_scats_func = params
        
    Temp = mQ/np.exp(x)
    
    GAM_QAD = GAMMA['Qad']
    GAM_QAAD = GAMMA['Qaad']
    
    GAM_TOT = sum(GAMMA[i] for i in GAMMA.keys())

        
            
        
    rhoRAD = (np.pi**2/30)*gstar_Q(Temp, mQ, 2.0)*Temp**4
    
    sRAD = ((2*np.pi**2)/(45))*gstarS_Q(Temp, mQ, 2.0)* Temp**3
    
    

    Ychi_eq = neq_gen_full(2.0, mQ, Temp)/sRAD
    
    Ya_eq = (45/(2*np.pi**4)) * (1/ gstarS_Q(Temp, mQ, 2.0))
    
    
    #rhoa = ga/30 * ((np.pi**(7/2))/(zeta(3)))**(4/3)*(na/ga)**(4/3)
        
    H =  np.sqrt((8/3) * np.pi * SC.GCF * (rhoRAD ))

    deltah = 1 + (1/3)*(Temp/gstarS_Q(Temp, mQ, 2.0))*SC.dgstarSdT(Temp)
    #print(fa)

    gamma_a = gam.dynamic_gamma_fun(mQ, poly_highT, mQ_scats_func, Temp) / fa**2

       
    S_Qad = 2*1*2/(2*np.pi**2) * (GAM_QAD + 2*GAM_QAAD) * mQ**2 *Temp*kn(1, np.exp(x))
    dYa =  (deltah/(H * sRAD)) * (S_Qad +  gamma_a*(1 - Ya/Ya_eq)) 
    
    return [dYa]


def eqs_FO_Y(x, y, params):
    ''' x : logx, Ychi: [Yield chi], Ya : [Yield ax] '''
    
    Ychi = y[0]
    Ya = y[1]
    rhor = y[2]
    
    mQ, GAMMA,  fa, poly_highT, mQ_scats_func = params

        
    Temp = mQ/np.exp(x)

    GAM_QAD = GAMMA['Qad']
    GAM_QAAD = GAMMA['Qaad']
    
    GAM_TOT = sum(GAMMA[i] for i in GAMMA.keys())
    
    GAM_QSM = GAM_TOT - GAM_QAD-GAM_QAAD
    
        
    rhoRAD = (np.pi**2/30)*gstar_Q(Temp, mQ, 2.0)*Temp**4
    
    sRAD = ((2*np.pi**2)/(45))*gstarS_Q(Temp, mQ, 2.0)* Temp**3
    
    
    
    Ychi_eq = neq_gen_full(2.0, mQ, Temp)/sRAD
    
    Ya_eq = (45/(2*np.pi**4)) * (1/ gstarS_Q(Temp, mQ, 2.0))
    
    
    #rhoa = ga/30 * ((np.pi**(7/2))/(zeta(3)))**(4/3)*(na/ga)**(4/3)
        
    H =  np.sqrt((8/3) * np.pi * SC.GCF * (rhoRAD ))

    deltah = 1 + (1/3)*(Temp/gstarS_Q(Temp, mQ, 2.0))*SC.dgstarSdT(Temp)
    gamma_a = gam.dynamic_gamma_fun(mQ, poly_highT, mQ_scats_func, Temp) / fa**2

    sgv = sigv(mQ)
    
    
    
    dYchi = deltah*((sgv*sRAD)/(H)) *(Ychi_eq**2 - Ychi * Ychi) + (kn(1, mQ/Temp)/kn(2,mQ/Temp))*(deltah/H) * (GAM_QAD*(Ya/Ya_eq)*Ychi_eq + GAM_QSM * Ychi_eq - GAM_TOT*Ychi)
    
    dYa =  deltah * (GAM_QAD/H) * (Ychi - (Ya/Ya_eq)*Ychi_eq) + ((deltah*gamma_a)/(H * sRAD)) * (1 - Ya/Ya_eq)
    
    rhoQ = np.sqrt(mQ**2 + 9*Temp**2)*Ychi*sRAD
    drhordt =  - 4* H*rhor + GAM_QSM * rhoQ + 2*np.sqrt(mQ**2 + 9*Temp**2) *sgv * sRAD**2*(Ychi * Ychi-Ychi_eq**2  )
    
    drhor= (deltah /H) *drhordt
    return [dYchi, dYa, drhor]



def eqs_after_FO(x, y, params):
    ''' x : u, y: [log(fQ), log(fa), log(fR)] '''
    lfQ = y[0]
    lfa = y[1]
    lfr = y[2]
    
    u = x
    
    Tin, rQ, ra, mQ, GAMMA = params
    
    
    rhoRADi = (np.pi**2/30)*gstar_Q(Tin, mQ, 2.0)*Tin**4
    
    rhoQi = rQ * rhoRADi
    
    rhoai = ra * rhoRADi 
    
    Temp = Tin*np.exp(lfr)*np.exp(-x)
    
    GAM_QAD = GAMMA['Qad']
    GAM_QAAD = GAMMA['Qaad']
    
    GAM_TOT = sum(GAMMA[i] for i in GAMMA.keys())
    
    GAM_QSM = GAM_TOT - GAM_QAD-GAM_QAAD 
        
    BRa = 0.5*GAM_QAD/GAM_TOT + 1/3 * GAM_QAAD/GAM_TOT 
    BRSM = 1-BRa
    
    rhoRAD = (np.pi**2/30)*gstar_Q(Temp, mQ, 2.0)*Temp**4
    
    sRAD = ((2*np.pi**2)/(45))*gstarS_Q(Temp, mQ, 2.0)* Temp**3
    
    
    rhoQ= rhoQi * np.exp(lfQ) * np.exp(-3*u)
    
    rhoa = rhoai * np.exp(lfa) * np.exp(-4*u)
    
    na = ((30*rhoa)/1)**(3/4) * ((1 * zeta(3))/np.pi**(7/2))

    na_rhoa= ((30)/1)**(3/4) * ((1 * zeta(3))/np.pi**(7/2)) * rhoa**(-1/4)
    
    na_eq = ni_eq_rel(1, Temp)



        
    H =  np.sqrt((8/3) * np.pi * SC.GCF * (rhoRAD + rhoQ + rhoa ))

    deltah = 1 + (1/3)*(Temp/gstarS_Q(Temp, mQ, 2.0))*SC.dgstarSdT(Temp)
        
    
    #print(BRSM, BRa, Temp)

    dlfQ = - GAM_TOT/H 
    dlfa = + (BRa*GAM_TOT*rhoQ)/(H * rhoa) 
    dlfr = 1 - 1/deltah + ( ( BRSM *  GAM_TOT) / ( 3 * H * Temp * sRAD * deltah ) ) * rhoQ 
    
    
    return [dlfQ, dlfa, dlfr]



#### run the code ##### 


def evolve_all(mQ_p, fa_p, lam_p, ratio_a = 1.0, ratio_Ti=1.0):

    Ti_bFO= mQ_p*ratio_Ti
    sRADi_bFO = ((2*np.pi**2)/(45))*np.vectorize(gstarS_Q)(Ti_bFO, mQ_p, 2.0)* Ti_bFO**3
    rhoRADi_bFO = (np.pi**2/30)*np.vectorize(gstar_Q)(Ti_bFO, mQ_p, 2.0)*Ti_bFO**4
    

    poly_highT_p, mQ_p_scats_func = gam.setup_gamma_Qscat_funcs(mQ_p)
    

    Tfin_bFO = mQ_p/100 # 10 Tsig 
    


    decay_dict = {'Qad': gamma_ad(mQ_p, fa_p, Lambda=lam_p ), 'QW3d': gamma_dW3(mQ_p, fa_p, Lambda=lam_p ),
                 'QBd' : gamma_dB(mQ_p, fa_p, Lambda=lam_p ), 'QuW': gamma_uW(mQ_p, fa_p, Lambda=lam_p ), 
                 'Qaad' : gamma_aad(mQ_p, fa_p, Lambda=lam_p ) }
    
    GAMTOT = sum(decay_dict[i] for i in decay_dict.keys())
    
    #print(decay_dict)
    
    TEND_val = fsolve(lambda x: SC.HubRad(x) - GAMTOT, mQ_p )
    #print(TEND_val, GAMTOT)
    if TEND_val > mQ_p/5:
        #print('here in equilib')


        params_input_eqdec = [mQ_p, decay_dict, fa_p,  poly_highT_p, mQ_p_scats_func]

        
        # sol_p1_eqdec = solve_ivp(lambda x, y : eqs_eqdecay_Y(x,y,params_input_eqdec ), 
        #                          [np.log(mQ_p/Ti_bFO), np.log(mQ_p/Tfin_bFO)],
        #                          y0=[ratio_a*Yeq_rel(1, Ti_bFO)], dense_output=True,
        #                          atol=1e-10, rtol=1e-10, method='BDF)
        sol_p1_eqdec = solve_ivp(lambda x, y : eqs_eqdecay_Y(x,y,params_input_eqdec ), 
                                 [np.log(mQ_p/Ti_bFO), np.log(mQ_p/Tfin_bFO)],
                                 y0=[ratio_a*Yeq_rel(1, Ti_bFO)])

        Temp_p1_eqdec= mQ_p/np.exp(sol_p1_eqdec.t)
        print(Temp_p1_eqdec)
        sRad_p1_eqdec = ((2*np.pi**2)/(45))*np.vectorize(gstarS_Q)(Temp_p1_eqdec, mQ_p, 2.0)* Temp_p1_eqdec**3

        #sRAD_p1_eqdec = ((2*np.pi**2)/(45))*gstarS_Q(Temp, mQ, gQ)* Temp**3
        na_eqdec = sRad_p1_eqdec *sol_p1_eqdec.y[0] 

        ngam_eqdec = (zeta(3)/np.pi**2) * 2.0 * Temp_p1_eqdec**3
        #nr_eqdec = (np.pi**2/30)*gstar_Q(Temp_p1_eqdec)*Temp_p1_eqdec**4

        

        rhoa_eqdec = (np.pi**2/30) * ((np.pi**2 * sRad_p1_eqdec *sol_p1_eqdec.y[0])/ zeta(3))**(4/3)

        
        rhor_eqdec = (np.pi**2/30)*np.vectorize(gstar_Q)(Temp_p1_eqdec, mQ_p, 2.0)*Temp_p1_eqdec**4
        
        rhor = rhor_eqdec
        rhoQ = neq_gen_full(2, mQ_p, Temp_p1_eqdec)
        rhoa = rhoa_eqdec

        Temp = Temp_p1_eqdec 

        #DNEFF = calc_deltaN(rhoa_eqdec[-1], rhor_eqdec[-1], Temp_p1_eqdec[-1] )
        
    
    if TEND_val <= mQ_p/5: 
        
        params_input_FO = [mQ_p, decay_dict,fa_p,  poly_highT_p, mQ_p_scats_func]
        
        
        
        if ratio_a==1.0:
            #print("here\n")
                                    
            sol_p1_bFO = solve_ivp(lambda x, y : eqs_FO_Y(x, y,
                                                          params_input_FO ),
                                   [np.log(mQ_p/Ti_bFO), np.log(mQ_p/Tfin_bFO)],
                                   y0=[neq_gen_full(2.0, mQ_p,Ti_bFO)/sRADi_bFO,ratio_a*neq_gen_full(1.0,mQ_p, Ti_bFO)/sRADi_bFO,rhoRADi_bFO], 
                                   dense_output=True, 
                               atol=1e-15, rtol=1e-13, method='BDF')
            
            
        else:

            try:

                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")  # Catch all warnings
                    sol_p1_bFO = solve_ivp(lambda x, y : eqs_FO_Y(x, y,
                                                                  params_input_FO ),
                                           [np.log(mQ_p/Ti_bFO), np.log(mQ_p/Tfin_bFO)],
                                           y0=[neq_gen_full(2.0, mQ_p, Ti_bFO)/sRADi_bFO,ratio_a*neq_gen_full(1.0, mQ_p, Ti_bFO)/sRADi_bFO], dense_output=True, 
                                           atol=1e-10, rtol=1e-10, method='LSODA')
                    # Check if any warnings were raised
                    if w:
                        for warn in w:
                            print(f"Warning raised with LSODA: {warn.message}")

                        sol_p1_bFO = solve_ivp(lambda x, y : eqs_FO_Y(x, y,
                                                                      params_input_FO ),
                                               [np.log(mQ_p/Ti_bFO), np.log(mQ_p/Tfin_bFO)],
                                               y0=[neq_gen_full(2.0, mQ_p,Ti_bFO)/sRADi_bFO,ratio_a*neq_gen_full(1.0,mQ_p, Ti_bFO)/sRADi_bFO], 
                                               dense_output=True, 
                                           atol=1e-10, rtol=1e-10, method='BDF')


            except Exception as e:
                print(f"Solver failed with exception: {e}")
                return np.nan



        #print(neq_gen_full(2.0, mQ_p, Ti_bFO)/sRADi_bFO,ratio_a*Yeq_rel(1.0, Ti_bFO))
        
        #print(sol_p1_bFO)

        Temp_sol_p1_bFO = mQ_p/np.exp(sol_p1_bFO.t)

        sRad_sol_p1_bFO = ((2*np.pi**2)/(45))*np.vectorize(gstarS_Q)(Temp_sol_p1_bFO, mQ_p, 2.0)* Temp_sol_p1_bFO**3

        nQ_bFO = sRad_sol_p1_bFO*sol_p1_bFO.y[0]
        

        na_bFO = sRad_sol_p1_bFO *sol_p1_bFO.y[1]

        rhor_bFO = (np.pi**2/30)*np.vectorize(gstar_Q)(Temp_sol_p1_bFO, mQ_p, 2.0)*Temp_sol_p1_bFO**4
        
        ngam_bFO = (zeta(3)/np.pi**2) * 2.0 * Temp_sol_p1_bFO**3
        


    
        ##################################
        ### Now solve for the decay ###### 
        ##################################


        Ti_aFO = Temp_sol_p1_bFO[-1]
        #print(Tfin_bFO , Temp_sol_p1_bFO[-1])

        print(Ti_aFO)

        rhoSMi_aFO = (np.pi**2/30)*np.vectorize(gstar_Q)(Ti_aFO, mQ_p, 2.0)*Ti_aFO**4
        #print("\n FO energy density %f \n" % (rhosig_sol_p1_bFO[-1]))
        
        
        
        rhoQ_bFO = mQ_p*nQ_bFO + 3/2 * nQ_bFO * Temp_sol_p1_bFO
        ratioQ_aFO= rhoQ_bFO[-1]/rhoSMi_aFO
        
        rhoa_bFO = (np.pi**2/30) * ((np.pi**2 * na_bFO)/ zeta(3))**(4/3)

        ratioa_aFO = rhoa_bFO[-1]/rhoSMi_aFO
        # ratioa_aFO = (np.pi**2/30)*Ti_bFO**4/((np.pi**2/30)*np.vectorize(gstar_Q)(Ti_bFO, mQ_p, 2.0)*Ti_bFO**4)
        



        ga_p  = 1.0


        #print(BRa_p, BRSM_p)
    
        #Tfin_aFO = 1e-4 # 10 Tsig 
        Tfin_aFO = max(TEND_val/100, 1e-4) # 10 Tsig 
        Rin_aFO = 1.0

        Rfin_aFO = 10**root(afin, [20.], args = (rhoQ_bFO[-1], rhoSMi_aFO+rhoa_bFO[-1], 1/GAMTOT, 0.0), method='lm', tol=1e-40).x * 1e1
        
        #Rfin_aFO = (gstarS_Q(Ti_aFO)/gstarS_Q(Tfin_aFO))**(1/3) * ((Ti_aFO*Rin_aFO)/(Tfin_aFO))

        uin_aFO, ufin_aFO = 0.0, np.log(Rfin_aFO/Rin_aFO) 

        params_input_p1_aFO = [Ti_aFO, ratioQ_aFO, ratioa_aFO, mQ_p, decay_dict]

        sol_p1_aFO = solve_ivp(lambda x, y : eqs_after_FO(x, y, params_input_p1_aFO ), [uin_aFO, ufin_aFO],
                               y0=[np.log(1.0), np.log(1.0), np.log(1.0)], atol=1e-10, rtol=1e-10, 
                              dense_output=True, method='LSODA')

        Temp_sol_p1_aFO = Ti_aFO*np.exp(sol_p1_aFO.y[2])*np.exp(-sol_p1_aFO.t)

        rhor_aFO = (np.pi**2/30)*np.vectorize(gstar_Q)(Temp_sol_p1_aFO, mQ_p, 2.0)*Temp_sol_p1_aFO**4
        rhoQ_aFO = ratioQ_aFO * rhor_aFO[0] * np.exp(sol_p1_aFO.y[0]) * np.exp(-3*sol_p1_aFO.t)
    

        sRad_sol_p1_aFO = ((2*np.pi**2)/(45))*np.vectorize(gstarS_Q)(Temp_sol_p1_aFO, mQ_p, 12)* Temp_sol_p1_aFO**3


        nQ_aFO = rhoQ_aFO/mQ_p

        rhoa_aFO = ratioa_aFO * rhor_aFO[0] * np.exp(sol_p1_aFO.y[1]) * np.exp(-4*sol_p1_aFO.t)
        
 
        rhor = np.concatenate((rhor_bFO, rhor_aFO), axis=0)

        rhoa = np.concatenate((rhoa_bFO, rhoa_aFO), axis=0)

        rhoQ = np.concatenate((rhoQ_bFO, rhoQ_aFO), axis=0)

        sRAD = np.concatenate((sRad_sol_p1_bFO, sRad_sol_p1_aFO), axis=0)


        Temp = np.concatenate((Temp_sol_p1_bFO ,Temp_sol_p1_aFO), axis=0)


        
        


        
    
    
    return rhor, rhoQ, rhoa, sRAD, Temp
