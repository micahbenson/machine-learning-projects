import numpy as np
import matplotlib.pyplot as plt

#Set iterations for simulation
N=10000

#Generate arrays to hold values 
a = np.zeros(N)
b = np.zeros(N)
e_out = np.zeros(N)

for i in range(N): 
    #Generate a dataset
    x1 = np.random.uniform(-1, 1)
    x2 = np.random.uniform(-1, 1)
    #Fit linear hypothesis for the dataset
    a[i] = x1+x2
    b[i] = -x1*x2
    #Calculate e_out for this hypothesis
    e_out[i] = ((x1+x2)**2 - 2*(-x1*x2))/3 + ((-x1*x2)**2) + 1/5

#Calculate average a across all the hypotheses in experiemnt to numerically estimate a
a_bar = np.mean(a)
#Calculate average b across all the hypotheses in experiment to numerically estimate b
b_bar = np.mean(b)
#Calculate the average e_out of hypotheses in experiment to numerically estimate e_out
e_out_bar = np.mean(e_out)

#Calculate bias using average function found from simulation
bias = ((a_bar)**2 - 2*(b_bar))/3 + ((b_bar)**2) + 1/5

#Calculate variance using variance of hypotheses in simulation
var_a = np.var(a)
var_b = np.var(b)
var = var_a/3 + var_b

#Print out findings
print(f'bias = {bias}')
print(f'variance = {var}')
print(f'bias + variance = {bias + var}')
print(f'e_out = {e_out_bar}')

#Use linspace for graphing average function
x = np.linspace(-1, 1)
g_bar = a_bar*x + b_bar

#Plot average function and target function
plt.plot(x, g_bar)
plt.plot(x, x**2)
plt.legend(['g_bar(x)', 'f(x)'])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Average Function Found by Simulations')
plt.show()