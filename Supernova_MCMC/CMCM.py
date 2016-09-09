###############################################
## James Cook
## MCMC Exercise
###############################################

print('')
print('###############################################')
print('Supernova Data MCMC paramter estimation')
print('###############################################')
print('')

###############################################
## Import Modules
###############################################

import numpy as np
from numpy.linalg import inv #To invert our covariant matrix
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import time
import argparse

print('Modules Loaded')

###############################################
## Define our Functions
###############################################

###############################################
## Flat Universe Equations
## z, Omega_m, h -> produces mu
###############################################

## Theoretical mu given h, Omega_m, and z
def mu(f_mu_h,f_mu_z,f_mu_omega_matter):

    f_mu_a = ((1.0)/(1.0 + f_mu_z))
    f_mu_s3 = ((1.0 - f_mu_omega_matter)/(f_mu_omega_matter))
    f_mu_s = f_mu_s3 ** ((1.0)/(3.0))

    f_mu_c = 3.0 * pow(10,5)
    f_mu_H0 = 100 # 100 h

    f_mu_eta = 2.0 * np.sqrt(f_mu_s3 + 1.0) * pow(( ((1.0)/(pow(f_mu_a,4.0))) - 0.1540 * ((f_mu_s)/(pow(f_mu_a,3.0))) \
    + 0.4304 * pow(((f_mu_s)/(f_mu_a)),2) + 0.19097 * ((pow(f_mu_s,3))/(f_mu_a)) + 0.066941 * pow(f_mu_s,4)), ((-1.0)/(8.0)))

    f_mu_a = 1

    f_mu_eta_1 = 2.0 * np.sqrt(f_mu_s3 + 1.0) * pow(( ((1.0)/(pow(f_mu_a,4.0))) - 0.1540 * ((f_mu_s)/(pow(f_mu_a,3.0))) \
    + 0.4304 * pow(((f_mu_s)/(f_mu_a)),2) + 0.19097 * ((pow(f_mu_s,3))/(f_mu_a)) + 0.066941 * pow(f_mu_s,4)), ((-1.0)/(8.0)))

    f_mu_Dl = ((f_mu_c)/(f_mu_H0)) * (1.0 + f_mu_z) * (f_mu_eta_1 - f_mu_eta)

    f_mu_result = 25.0 - 5.0 * np.log10(f_mu_h) + 5.0 * np.log10(f_mu_Dl)

    return f_mu_result

print('Flat Universe Equations Defined')

###############################################
## MCMC
###############################################

def likelihood(f_l_h, f_l_omega_matter, f_l_inverted_matrix, f_l_z_data, f_l_mu_data):

    f_l_calc=np.empty(31)

    for like_q in range(0,31):
        f_l_calc[like_q] = (f_l_mu_data[like_q] - mu(f_l_h, f_l_z_data[like_q], f_l_omega_matter))

    f_l_result = (-0.5*np.dot(f_l_calc,np.dot(f_l_inverted_matrix,f_l_calc))) - (pow((f_l_h-0.738),2.0)/(2*0.024*0.024)) #Second term is from Hubble

    return (f_l_result)

def MCMC(f_MCMC_N, f_MCMC_var, f_MCMC_inverted_matrix, f_MCMC_z_data, f_MCMC_mu_data):

    MCMC_var=f_MCMC_var

    MCMC_counter=0

    MCMC_result =[]
    MCMC_h_result=[]
    MCMC_o_result=[]
    MCMC_h_chain=[]
    MCMC_o_chain=[]

    for MCMC_MAIN in range(0,f_MCMC_N):
        if MCMC_MAIN ==0:
            f_MCMC_o_start = np.random.uniform(0.1,0.9,1)[0]
            f_MCMC_h_start = np.random.uniform(0.1,0.9,1)[0]
            MCMC_0 = likelihood(f_MCMC_h_start, f_MCMC_o_start, f_MCMC_inverted_matrix, f_MCMC_z_data, f_MCMC_mu_data)

        f_MCMC_o1 = np.random.normal(f_MCMC_o_start,MCMC_var)
        while f_MCMC_o1<0.1 or f_MCMC_o1>0.95:
            f_MCMC_o1 = np.random.normal(f_MCMC_o_start,MCMC_var)

        f_MCMC_h1 = np.random.normal(f_MCMC_h_start,MCMC_var)
        while f_MCMC_h1<0.1 or f_MCMC_h1>0.95:
            f_MCMC_h1 = np.random.normal(f_MCMC_h_start,MCMC_var)

        MCMC_1 = likelihood(f_MCMC_h1, f_MCMC_o1, f_MCMC_inverted_matrix, f_MCMC_z_data, f_MCMC_mu_data)

        MCMC_k=np.random.uniform(0,1,1)[0]

        if MCMC_k<min(1.0,np.exp(MCMC_1-MCMC_0)):
            MCMC_counter=MCMC_counter+1
            f_MCMC_o_start = f_MCMC_o1
            f_MCMC_h_start = f_MCMC_h1
            MCMC_0 = MCMC_1
        else:
            MCMC_counter=MCMC_counter+0

        if MCMC_MAIN>200:
            MCMC_o_result.append(f_MCMC_o_start)
            MCMC_h_result.append(f_MCMC_h_start)

        if args.con==False:
            if MCMC_MAIN%200==0 and MCMC_MAIN>0:
                print str((MCMC_MAIN/float(f_MCMC_N))*100.0),'%'

        MCMC_o_chain.append(f_MCMC_o_start)
        MCMC_h_chain.append(f_MCMC_h_start)

    MCMC_result.append(MCMC_o_result)
    MCMC_result.append(MCMC_h_result)
    MCMC_result.append(MCMC_o_chain)
    MCMC_result.append(MCMC_h_chain)
    MCMC_result.append((MCMC_counter))

    return MCMC_result

def Gelman_Rubin(f_G_R):

    total_chain_mean_o =[]
    total_chain_mean_h = []
    total_chain_var_o =[]
    total_chain_var_h =[]

    Boa =[]
    Bha = []


    for i in range(0,len(f_G_R)):
        o_chain_mean = np.mean(f_G_R[i][0])
        h_chain_mean = np.mean(f_G_R[i][1])
        total_chain_mean_h.append(h_chain_mean)
        total_chain_mean_o.append(o_chain_mean)

        total_chain_var_o_1 =[]
        total_chain_var_h_1 =[]

        for q in range(0,len(f_G_R[i][0])):
            o_q = pow((f_G_R[i][0][q]-total_chain_mean_o[i]),2)
            h_q = pow((f_G_R[i][1][q]-total_chain_mean_h[i]),2)

            total_chain_var_o_1.append(o_q)
            total_chain_var_h_1.append(h_q)

        total_chain_var_o.append(np.mean(total_chain_var_o_1))
        total_chain_var_h.append(np.mean(total_chain_var_h_1))

    for j in range(0,len(f_G_R)):
        Boa.append(total_chain_mean_o[j] - np.mean(total_chain_mean_o))
        Bha.append(total_chain_mean_h[j] - np.mean(total_chain_mean_h))

    BO = ((len(f_G_R[0][0]))/(len(f_G_R)-1.)) * np.sum(Boa)
    BH = ((len(f_G_R[0][1]))/(len(f_G_R)-1.)) * np.sum(Bha)

    WH = np.mean(total_chain_var_h)
    WO = np.mean(total_chain_var_o)

    VH = ((len(f_G_R[0][0]) -1.)/(len(f_G_R[0][0]))) * WH + ((len(f_G_R) + 1.)/(len(f_G_R)*(len(f_G_R[0][0]))))*BH
    VO = ((len(f_G_R[0][0]) -1.)/(len(f_G_R[0][0]))) * WO + ((len(f_G_R) + 1.)/(len(f_G_R)*(len(f_G_R[0][0]))))*BO

    GR_return_1=[VH/WH,VO/WO]
    return GR_return_1

print('MCMC Defined')

print('Entering Main')

###############################################
## Main
###############################################

parser = argparse.ArgumentParser(description='Supernova Data Analysis')
parser.add_argument("--plot",action="store_true",help='Plots results of MCMC')
parser.add_argument("--con",action="store_true",help='Check convergence of MCMC')
args = parser.parse_args()

## Import the Data
input_data = np.genfromtxt('jla_mub.txt', delimiter=' ', skip_header=0, skip_footer=0, names=['z', 'mu'])

print('Data imported')

#Load the Covariant Matrix
input_covariant_matrix = np.genfromtxt('jla_mub_covmatrix.txt', delimiter=' ', skip_header=0, skip_footer=0)
input_covariant_matrix=input_covariant_matrix.reshape(31,31)

print('Matrix imported')

## Invert the Covariant Matrix
inverted_input_covariant_matrix = inv(input_covariant_matrix)

print('Matrix inversed')

array_mu =[]

if args.con:
    print 'Beginning the Gelman-Rubin convergence test'

    GR =[]
    GR1 =[]
    GR2=[]

    GR.append(MCMC(500,0.015,0.7,0.3,inverted_input_covariant_matrix, input_data['z'], input_data['mu']))
    GR.append(MCMC(500,0.015,0.7,0.3,inverted_input_covariant_matrix, input_data['z'], input_data['mu']))

    GR2.append(MCMC(1000,0.015,0.7,0.3,inverted_input_covariant_matrix, input_data['z'], input_data['mu']))
    GR2.append(MCMC(1000,0.015,0.7,0.3,inverted_input_covariant_matrix, input_data['z'], input_data['mu']))
    GR1.append(Gelman_Rubin(GR))
    GR1.append(Gelman_Rubin(GR2))

    print GR1

    exit()
    GRH=[]
    GRO=[]
    for con_z in range(0,len(GR1)):
        GRH.append(GR1[con_z][0])
        GRO.append(GR1[con_z][1])

    plt.figure('h')
    plt.plot(range(0,len(GRO)),GRO)
    plt.ylim([0,1])
    plt.show()
    print 'Exiting'
    exit()

print('MCMC beginning')

start_time = time.time()

main_run_length = 2000

main_results=MCMC(main_run_length,0.015,inverted_input_covariant_matrix, input_data['z'], input_data['mu'])

print('MCMC completed')

print('')
print('###############################################')
print('Results')
print('###############################################')
print('')
print "Ran in",int(time.time() - start_time), "seconds"
print 'Acceptance Ratio',((main_results[4]*100)/main_run_length),'%'
print('')
print('Omega_m')
print str(np.mean(main_results[0]))
print str(np.var(main_results[0]))
print('')
print('h')
print str(np.mean(main_results[1]))
print str(np.var(main_results[1]))
print('')

if args.plot==False:
    print 'Exiting'
    exit()

plt.figure('Omega_m')
plt.title('$\Omega_m$')
plt.hist(main_results[0],bins=100, normed=True)
x = np.linspace(min(main_results[0]), max(main_results[0]), 100)
plt.plot(x,mlab.normpdf(x, np.mean(main_results[0]), np.sqrt(np.var(main_results[0]))))
plt.xlabel('$\Omega_m$')
plt.ylabel('Frequency')

plt.figure('O_chain')
plt.title('$\Omega_m chain$')
plt.plot(range(0,len(main_results[2])),main_results[2])
plt.xlabel('Steps')
plt.ylabel('$\Omega_m$')

plt.figure('h')
plt.title('$h$')
plt.hist(main_results[1],bins=100, normed=True)
x = np.linspace(min(main_results[1]), max(main_results[1]), 100)
plt.plot(x,mlab.normpdf(x, np.mean(main_results[1]), np.sqrt(np.var(main_results[1]))))
plt.ylabel('Frequency')
plt.xlabel('$h$')
plt.ylabel('Frequency')

plt.figure('h_chain')
plt.title('$h chain$')
plt.plot(range(0,len(main_results[3])),main_results[3])
plt.xlabel('Steps')
plt.ylabel('$h$')

plt.figure('hmhist')
plt.hist2d(main_results[0],main_results[1], bins=1000)
plt.title('$h$ vs $\Omega_m$')
plt.xlabel('$h$')
plt.xlabel('$\Omega_m$')

plt.show()

print 'Exiting'
exit()

#Loop to calculate theoretical values
#for i_loop in range(0,len(input_data['z'])):
#    array_mu.append(mu(0.7,input_data['z'][i_loop],0.3))
