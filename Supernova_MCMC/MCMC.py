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
from numpy.linalg import inv
import matplotlib
import matplotlib.pyplot as plt
import math
import time
from decimal import Decimal

print('Modules imported')

###############################################
## Define our Functions
###############################################

###############################################
## Flat Universe Equations
## z, Omega_m, h -> produces mu
###############################################

## Calculate Eta given z and Omega_m
def eta(z,Omega_m):

    s3 = (1.0 - Omega_m)/Omega_m
    s  = s3 ** (1.0/3.0)
    a = 1.0/(1.0 + z)

    eta_result = 2.0 * math.sqrt(s3 + 1.0) * math.pow(( (1.0/math.pow(a,4.0)) - 0.1540 * (s/math.pow(a,3.0)) + \
     0.4304 * math.pow((a/s),2.0) + 0.19097 * (math.pow(s,3.0)/a) + 0.066941 * math.pow(s,4.0)),((-1.0)/8.0))

    return eta_result

## Calculate Luminosity Distance (h=1) (Valid in a flat universe) given z and Omega_m
def DL(z, Omega_m):

    c = 3*math.pow(10,3)
    H0 = 1.

    DL_result = (c/H0) * (1.0 + z) * ( eta(0,Omega_m) - eta(z,Omega_m))

    return DL_result

## Calculate mu given Dl and h
def mu(Dl, h):

    mu_result = 25 - 5 * np.log10(h) + 5 * np.log10(Dl)

    return mu_result

## Function to do all of the calculations and return mu
def all(z,Omega_m,h):

    return mu(DL(z,Omega_m), h)

###############################################
## MCMC
###############################################

def likelihood(h, Omega_m, Covariance_Matrix_Inverted, z_input_data, mu_input_data):

    likelihood_sum=[]

    for i in range(0,31):
        for j in range(0,31):

            likelihood_i_j = (mu_input_data[i] - all(z_input_data[i],Omega_m,h)) * Covariance_Matrix_Inverted[i][j] * (mu_input_data[j] - all(z_input_data[j],Omega_m,h))
            likelihood_sum.append(likelihood_i_j)
            likelihood_i_j

    L = ((-1.0)/(2.0)) * np.sum(likelihood_sum)

    return L

def MCMC_main(trial_length,trial_N,Omega_m_min,Omega_m_max,h_min,h_max, varh, varo, Covariance_Matrix_Inverted, z_input_data, mu_input_data):

    o_comp=[]
    h_comp=[]
    total=[]
    for i in range(0,trial_N):
        o_start = np.random.uniform(Omega_m_min,Omega_m_max,1)[0]
        h_start = np.random.uniform(h_min,h_max,1)[0]
        o1 = np.random.normal(o_start,varo)
        h1 = np.random.normal(h_start,varh)

        o_current = o_start
        h_current = h_start
        o_trial = o1
        h_trial = h1
        counter=0.
        for j in range(0,trial_length):

            if j!=0:
                o_trial = np.random.normal(o_current,varo)
                h_trial = np.random.normal(h_current,varo)

            if o_trial<Omega_m_min or o_trial>Omega_m_max or h_trial<h_min or h_trial>h_max:
                #print('Outside')
                False
            else:
                #print('Inside')
                lc = likelihood(h_current, o_current, Covariance_Matrix_Inverted, z_input_data, mu_input_data)
                lt = likelihood(h_trial, o_trial, Covariance_Matrix_Inverted, z_input_data, mu_input_data)
                #print np.exp((lt)-(lc))
                #print lt/lc

                #print lc
                #print lt
                #print np.exp(lc)

                #print np.exp(lt-lc)

                k=np.random.uniform(0,1,1)[0]

                if lt>lc:
                    if k<0.5:
                    #print 'hit'
                        counter=counter+1.
                        o_current=o_trial
                        h_current=h_trial

            o_comp.append(o_current)
            h_comp.append(h_current)
        #print counter

        #print o_current
        #print h_current

        #if i%10==0:
        #    print i

        total.append(o_comp)
        total.append(h_comp)

    return total


print('Functions defined')

###############################################
## Main
###############################################

## Import the Data
input_data = np.genfromtxt('jla_mub.txt', delimiter=' ', skip_header=0, skip_footer=0, names=['z', 'mu'])
z_array = input_data['z']
mu_array = input_data['mu']

print('Data imported')

input_covariant_matrix = np.genfromtxt('jla_mub_covmatrix.txt', delimiter=' ', skip_header=0, skip_footer=0)
input_covariant_matrix=input_covariant_matrix.reshape(31,31)
print('Covariant Matrix List Imported')

## Invert the Covariant Matrix
inverted_input_covariant_matrix = inv(input_covariant_matrix)
print('Matrix Inverted')

## Seed our random number generator
random_seed=int(time.clock()*1000000000)
np.random.seed(random_seed)

print('Random Number Generator Seeded')
print('')

result = MCMC_main(10000,2,0.1,0.9,0.1,0.9,1,1,inverted_input_covariant_matrix, z_array, mu_array)
#print(likelihood(0.7, 0.3, input_covariant_matrix, z_array, mu_array))
#print result
plt.plot(range(0,len(result[0])), result[0],label='o')
plt.plot(range(0,len(result[0])), result[1],label='h')
plt.plot(range(0,len(result[0])), result[2],label='o')
plt.plot(range(0,len(result[0])), result[3],label='h')
#plt.ylabel('y')
#plt.legend(loc='lower right')
#plt.subplot(2, 1, 2)
#plt.plot(input_data['z'], gaussian_random_array, label='Gaussian')
#plt.errorbar(input_data['z'],gaussian_random_array,0.1,0.0, ls='none')
#plt.xlabel('z')
#plt.ylabel('mu')
#plt.legend(loc='lower right')
plt.show()

print('')
print('Program Complete')
exit()

plt.hist2d(result[0],result[1])
plt.xlabel('$\Omega_m$')
plt.ylabel('$h$')
plt.xlim([0,1])
plt.ylim([0,1])
plt.colorbar()
plt.show()
#print(likelihood(0.7, 0.3, inverted_input_covariant_matrix, z_array, mu_array))
#print(likelihood(0.7, 0.31, inverted_input_covariant_matrix, z_array, mu_array))

print('')
print('Program Complete')
exit()

###############################################
## Legacy Code from prelimnary Exercise
###############################################

# Calculate our thoretical values
theory_mu_2_array = []
#theory_mu_3_array = []
#theory_mu_4_array = []
#theory_mu_5_array = []
#gaussian_random_array = []

for i in range(0,len(z_array)):
    theory_mu_2_array.append(all(z_array[i],0.3,0.7))
#    theory_mu_3_array.append(all(z_array[i],0.3))
#    theory_mu_4_array.append(all(z_array[i],0.4))
#    theory_mu_5_array.append(all(z_array[i],0.5))
#    gaussian_random_array.append(np.random.normal(all(z_array[i],0.3),0.1))


# Plot everything
plt.figure('Distance Modulus vs Redshift')
#plt.subplot(2, 1, 1)
plt.title('Distance Modulus vs Redshift')
plt.plot(input_data['z'], theory_mu_2_array, label='0.2')
#plt.plot(input_data['z'], theory_mu_3_array, label='0.3')
#plt.plot(input_data['z'], theory_mu_4_array, label='0.4')
#plt.plot(input_data['z'], theory_mu_5_array, label='0.5')
plt.plot(input_data['z'], input_data['mu'],ls='none',marker='x', label='Observation')
plt.ylabel('mu')
plt.legend(loc='lower right')
#plt.subplot(2, 1, 2)
#plt.plot(input_data['z'], gaussian_random_array, label='Gaussian')
#plt.errorbar(input_data['z'],gaussian_random_array,0.1,0.0, ls='none')
#plt.xlabel('z')
#plt.ylabel('mu')
#plt.legend(loc='lower right')
plt.show()
