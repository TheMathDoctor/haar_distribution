import unitary_stuffs as u
import numpy as np
pi = np.pi
class haar_distribution:

  def __init__(self,dimension):
    self.dimension = dimension #Dimension of the vector space on which the unitaries act
  #unnormalized haar density. 
  #Lambda is a dxd matrix of values between 0 and 2pi
  def density(self,Lambda):
    d = self.dimension
    value = 1
    for M in range(d-1):
      for N in range(M+1,d):
        value = value*np.sin(Lambda[M,N])*(np.cos(Lambda[M,N]))**(2*(N-M)-1)
    return value
  
  #normalized haar density
  def normalized_density(self,Lambda):
    d = self.dimension
    numerator = (2*pi)**(d*(d+1)*.5)
    denominator = 1
    for m in range(1,d):
        for n in range(m+1,d+1):
            denominator = denominator*2*(n-m)
    
    normalization_constant = numerator/denominator
    return self.density(Lambda)/normalization_constant
  #generates a sample of approximately uniformly distributed unitaries 
  def generate_sample(self,size, burn_in):
    #burn_in is expected to be a positive number less than 1. Usual is around .2. 
    iterations = int(size/(1-burn_in))
    d = self.dimension
    Lambdas = []
    initial_Lambda = self.proposal(d)
    Lambdas.append(initial_Lambda)

    for N in range(iterations):
      w = self.proposal(d)
      alpha = self.a(Lambdas[N],w)
      flip = self.coin(alpha)

      if flip == 1:
        Lambdas.append(w)
      else:
        Lambdas.append(Lambdas[N])

    
    Lambda_Sample = Lambdas[-size:]
    Unitary_Sample = []
    for L in Lambda_Sample:
      Unitary_Sample.append(u.construct_unitary(d,L))
    
    return Unitary_Sample
    
  #### probability helper methods for MCMC ####
  
  #acceptance probability for metropolis hastings algorithm
  def a(self,v,w):
    d = self.dimension
    M = [1,self.density(w)/self.density(v)]
    return min(M)
   
  @staticmethod
  def coin(p):
    return np.random.binomial(1,p)
    
  @staticmethod
  #uniform proposal method for metropolis hastings algorithm
  def proposal(d):
    P = np.zeros((d,d))
    for M in range(d):
        for N in range(d):
            if M<N:
                P[M,N]=np.random.uniform()*pi/2
            elif M==N:
                P[N,M]=np.random.uniform()*pi*2
            else:
                P[M,N]=np.random.uniform()*pi
  
    return P
    
  
d =4
#This lets you interect with the unnormalized and normalized density.
example = haar_distribution(d)
Lambda = np.zeros((d,d)) 
value = example.density(Lambda)
print(value)

Lambda = np.ones((d,d))
value = example.normalized_density(Lambda)
print(value)
 
#generate a sample
burn_in = .2
size = 100
sample = example.generate_sample(size,burn_in)
L = len(sample)
print(L)
 
#example haar integral approximation
d = 2
distribution = haar_distribution(d)
burn_in = .2
size = 1000
sample = distribution.generate_sample(size, burn_in)
X = [[1,0],[0,1]]
integral = np.zeros((2,2))
for U in sample:
    integral = integral + np.matmul(np.matmul(U,X),np.transpose(np.conjugate(U)))

integral = (1/size)*integral
error = 0
print(integral)
I = np.identity(2)
for i in range(2):
    for j in range(2):
        error += np.abs(integral[i,j]-I[i,j])**2
hs_error = np.sqrt(error)
print("error is ", hs_error)
