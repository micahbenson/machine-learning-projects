import numpy as np 
import matplotlib.pyplot as plt

#PROBLEM 2
#Create a function that returns a vector 
def vector():
    w = np.zeros(11)
    for i in range(10): 
        w[i+1] = np.random.uniform(0, 1)
    return w

#A function that returns sample of 100 points from {-1, 1} uniform distribution
def sample(): 
    #Generate 100 11dimension points from uniform {0, 1} distributiongit 
    x = np.random.randint(2, size=(11, 100))
    #Replace 0s with -1s to make uniform {-1, 1} distribution
    x[x==0] = -1
    #Replace the first dimension of each point with 1 to fit ML conventions, leaving 10d points
    x[0, :] = 1
    return x

#Function to classify a sample of points by a vector w 
#x: a sample of points 
#w: a vector used to classify the points
def classify(x, w): 
    return np.sign(np.dot(w, x))

#Function to calculate R to be used in bound on iterations
#x: a sample of points
def R(x): 
    x2 = np.square(x)
    sums = np.sum(x2, axis=0)
    return np.max(sums)**.5

#Function to calculate rho to be used in bound on iterations
#x: a sample of points 
#w: vector used to classify points
def rho(x, w):
    y = classify(x, w)
    wx = np.dot(w, x)
    return np.min(y*wx) 

#Function to calculate the bound on iterations
#x: a sample of points 
#w: vector used to classify points
def bound(x, w): 
    r = R(x) 
    p = rho(x, w)
    dot = np.dot(w, np.transpose(w))
    return (r**2)*(dot)/(p**2)

#Initialize array to hold bound calculation for 1000 simulations
bound_array = np.zeros(1000)
#Initialize array to hold iterations of PLA in 1000 simulations
iterations = np.zeros(1000) 
#Loop to run simulation 1000 times
for i in range(1000): 
    #Create w_star 
    w_star = vector()
    #Create sample of 100 points
    x = sample()
    #Classify sample according to w_star
    bound_array[i] = bound(x, w_star)
    #Classify sample according to w_star
    y = classify(x, w_star)
    #Initialize a zero weight vector for start of PLA
    w = np.zeros(11)
    #Classify sample according to w 
    y_hat = classify(x, w)
    #Repeat loop while classification by w does does not equal classification by true w
    while(np.array_equal(y_hat, y)==False):
        #Create array of misclassified points
        mc = x[:, y_hat != y]
        #Create parallel array of classifications for the misclassified points
        mcy = y[y_hat != y]
        #Choose a random index in the miscalssified arrays 
        choice = np.random.randint(len(np.transpose(mc)))
        #Update w to be w plus the misclassified point chosen by the index multiplied by its classification 
        w += mc[:, choice]*mcy[choice]
        #Classify sample according to the updated w
        y_hat = classify(x, w)
        #Add one to PLA iterations for this simulation
        iterations[i] += 1

#Create histogram of iterations
plt.hist(iterations, bins=30)
plt.xlabel('Number of Iterations of PLA Algorithm')
plt.ylabel('Frequency')
plt.title('Iterations of PLA Algorithm for 1000 Simulations')
plt.show()

#Create histogram of log(bound-iterations)
log_dif = np.log(bound_array-iterations) 
plt.hist(log_dif, bins=30)
plt.xlabel('log(Bound on Iterations - Iterations)')
plt.ylabel('Frequency')
plt.title('Comparison of Bound on PLA Iterations to Iterations in Simulation')
plt.show()

#PROBLEM 3
x = np.linspace(0, 1)
#function for bound 
y = 4*np.exp((-2*6)*x**2)
#funciton for error 
z = 1-(1-(1-x)**6)**2
plt.plot(x, y)
plt.plot(x, z)
plt.legend(['Bound', 'Pr[max{|vi - µi|}i=1,...,M > ε]'])
plt.xlabel('Value of ε')
plt.ylabel('Probability')
plt.title('Pr[max{|vi - µi|}i=1,...,M > ε] and its Hoeffding bound')
plt.show()