###############################################
## James Cook
## Preliminary Exercise
###############################################

###############################################
## Import Modules
###############################################

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import time

def eta(z,Omega_m):

    s3 = (1.0 - Omega_m)/Omega_m
    s  = s3 ** (1.0/3.0)
    a = 1.0/(1.0 + z)

    eta_result = 2.0 * math.sqrt(s3 + 1.0) * math.pow(( (1.0/math.pow(a,4.0)) - 0.1540 * (s/math.pow(a,3.0)) + \
     0.4304 * pow((a/s),2.0) + 0.19097 * (math.pow(s,3.0)/a) + 0.066941 * math.pow(s,4.0)),((-1.0)/8.0))

    return eta_result

def DL(z,Omega_m):

    c = 3*math.pow(10,5)
    H0 = 70.0

    DL_result = (c/H0) * (1.0 + z) * ( eta(0,Omega_m) - eta(z,Omega_m))

    return DL_result

def mu(Dl):

    h = 70.0/100.0

    mu_result = 25 - 5 * math.log10(h) + 5 * math.log10(h*Dl)

    return mu_result

def all(z,Omega_m):

    return mu(DL(z,Omega_m))

## Import the Data
input_data = np.genfromtxt('jla_mub.txt', delimiter=' ', skip_header=0, skip_footer=0, names=['z', 'mu'])

z_array = input_data['z']
mu_array = input_data['mu']

## Seed our random number generator
random_seed=int(time.clock()*1000000000)
np.random.seed(random_seed)

## Calculate our thoretical values
theory_mu_2_array = []
theory_mu_3_array = []
theory_mu_4_array = []
theory_mu_5_array = []
gaussian_random_array = []

for i in range(0,len(z_array)):
    theory_mu_2_array.append(all(z_array[i],0.2))
    theory_mu_3_array.append(all(z_array[i],0.3))
    theory_mu_4_array.append(all(z_array[i],0.4))
    theory_mu_5_array.append(all(z_array[i],0.5))
    gaussian_random_array.append(np.random.normal(all(z_array[i],0.3),0.1))


## Plot everything
plt.figure('Distance Modulus vs Redshift')
plt.subplot(2, 1, 1)
plt.title('Distance Modulus vs Redshift')
plt.plot(input_data['z'], theory_mu_2_array, label='0.2')
plt.plot(input_data['z'], theory_mu_3_array, label='0.3')
plt.plot(input_data['z'], theory_mu_4_array, label='0.4')
plt.plot(input_data['z'], theory_mu_5_array, label='0.5')
plt.plot(input_data['z'], input_data['mu'],ls='none',marker='x', label='Observation')
plt.ylabel('mu')
plt.legend(loc='lower right')
plt.subplot(2, 1, 2)
plt.plot(input_data['z'], gaussian_random_array, label='Gaussian')
plt.errorbar(input_data['z'],gaussian_random_array,0.1,0.0, ls='none')
plt.xlabel('z')
plt.ylabel('mu')
plt.legend(loc='lower right')
plt.show()

exit()
