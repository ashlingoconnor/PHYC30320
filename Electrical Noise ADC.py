#!/usr/bin/env python
# coding: utf-8

# In[20]:


from pydaqmx_helper.adc import ADC


# In[21]:


myADC = ADC()


# In[22]:


myADC.addChannels([3], minRange=-5, maxRange=5)


# In[23]:


data = myADC.sampleVoltages(10,10000)
data2 = myADC.sampleVoltages(10000,10000)


# In[26]:


#Rate of 10 counts at 10000Hzs-1 gives a time interval of 1ms


# In[103]:


sample3 = data[3]
sample23 = data2[3]


# In[5]:


np.savetxt("sample3.txt", sample3)
np.savetxt("sample23.txt", sample23)


# In[9]:


np.savetxt("poisson_data.txt", poisson_data)


# In[24]:


a = np.loadtxt("sample23.txt")
print(a)


# In[104]:


print(sample3)


# In[14]:


import numpy as np


# In[15]:


import matplotlib.pyplot as plt


# In[19]:


a = np.loadtxt("sample23.txt")

plt.hist(a, bins =200)
plt.title("Distribution of Voltages")
plt.xlabel("Voltage")
plt.ylabel("Frequency");
plt.show()

plt.savefig("voltage_frequency.pdf")

#At a rate of 10000 counts per second, the data looks like Poisson's Distribution


# In[20]:


a = np.loadtxt("sample23.txt")

plt.plot(a)
plt.title("Direct Noise Sampling")
plt.xlabel("Readings")
plt.ylabel("Frequency");
plt.show()

plt.savefig("noise_sample.pdf")

#Amplitude of 5V


# In[25]:


mean_exp = np.mean(a)
print(np.mean(a))

#Experimental Mean = 2.514146006876733


# In[53]:


standard_deviation_exp = np.std(a)
print(np.std(a))

import statistics
variance_exp = statistics.variance(a)
print(statistics.variance(a))

#From theory, variance = Standard De. squared
variance_exp_2 = (np.std(a))**2
print(variance_exp_2)


# In[168]:


#Formula to find Variance from King Poisson's Statistics Paper

for x in np.array(data2[3]):
    mean_difference = x - mean

variance_1 = np.sum(10000*(mean_difference)**2)/(10000)

print(variance_1)

#Variance = 0.6966522570381021
#Standard deviation = Square root of the Variance

standard_deviation_1 = np.sqrt(variance_1)
print(standard_deviation_1)

#Standard deviation = 1.5856058800587027


# In[160]:


#In theory the variance = mean and the standard deviation = square root of both
#This theory is not satisfied here


# In[7]:


#Generating Poisson's Distribution

from scipy.stats import poisson

poisson_data = []

y = poisson.rvs(mu=2.514146006876733, size=10000)
poisson_data.append(y)

print(y)


# In[23]:


x = y
plt.hist(x, bins =100)
plt.title("Theoretical Poisson's Distribution")
plt.xlabel("Voltages")
plt.ylabel("Frequency");
plt.show()

plt.savefig("poissons_data.pdf")

#Histogram of Poisson's theoretical Distribution from this mean
#Low mean suggests an unsymmetrical distribution


# In[36]:


#Poisson Probability Mass Function based off this mean

def poisson_pmf(x,mean):
    return ((2.514146006876733**(x))*(np.exp(-2.514146006876733)))/(np.math.factorial(int(x)))

sample_data = []
#Creating empty set to fill

for x in np.array(sample23):
    y = poisson_pmf(x,mean)
    sample_data.append(y)
    print(y)


#Fill the empty set with y values as they come

sample_data = np.asarray(sample_data)

#Turn this set into a numpy array, easier to plot


# In[62]:


mean_theor = np.mean(poisson_data)
print(np.mean(poisson_data))

#Theoretical Standard Deviation = Square Root of Mean 
standard_deviation_theor = np.std(poisson_data)
print(standard_deviation_theor)

#Theoretical Variance = Standard Deviation Squared 
variance_theor = (standard_deviation_theor)**2
print((standard_deviation_theor)**2)


# In[66]:


#Calculating Chi-Squared Critical Value

import scipy.stats

scipy.stats.chi2.ppf(q=0.95, df=10000)


# In[82]:


observed = a
expected = poisson_data

print(expected)
print(((observed-expected)**2)/(expected))

chi_squared_stat = (((observed-expected)**2)/expected).sum()


# In[79]:


p_value = 1 - scipy.stats.chi2.cdf(x=chi_squared_stat, df=10000)
print(p_value)
print(scipy.stats.chi2.cdf(x=chi_squared_stat, df=10000))


# In[75]:


#Chi Squared Test

exp_theory_diff = []

while i< 10000:
    y = poisson_data - a
    exp_theory_diff.append(y)
    i += 1

def chi_squared(exp_theory_diff, poisson_data):
    return np.sum((exp_theory_diff**2)/poisson_data)

print(chi_squared(exp_theory_diff, poisson_data))

#Calculated Value is higher than Critical Value


# In[65]:


scipy.stats.chisquare(f_obs= a, f_exp= poisson_data)


# In[209]:


#Experimental Gaussian Distribution from King Poisson's 

def Gauss_distr(numerator, standard_deviation_1):
    return (numerator)/(standard_deviation_1)*(np.sqrt(2*np.math.pi))

gaussian_exp_data = []

for x in np.array(a):
    mean_difference = x - mean_exp
    numerator = np.exp((-1/2)*((mean_difference)/(standard_deviation_1))**2)
    z = Gauss_distr(numerator, standard_deviation_1)
    gaussian_exp_data.append(z)
    
gaussian_exp_data = np.asarray(gaussian_exp_data)


# In[46]:


from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns

sns.distplot(random.normal(loc=2.517426351121999, scale= 0.33709705587597266, size=10000), hist=True)

plt.show()

plt.savefig("Gauusian_Dist.pdf")

#Expected Gaussian distribution from Experimental Results


# In[217]:


#Theoretical Gaussian Distribution from King Poisson's 

def Gauss_distr_theor(numerator_2, standard_deviation_2):
    return (numerator_2)/(standard_deviation_2)*(np.sqrt(2*np.math.pi))

gaussian_exp_data_2 = []

for x in np.array(sample_data):
    mean_difference_2 = x - mean_theor
    numerator_2 = np.exp((-1/2)*((mean_difference_2)/(standard_deviation_2))**2)
    w = Gauss_distr_theor(numerator_2, standard_deviation_2)
    gaussian_exp_data_2.append(w)
    
gaussian_exp_data_2 = np.asarray(gaussian_exp_data_2)


# In[54]:


#Fraction of Distribution of Experimental Data

positive = mean_exp + standard_deviation_exp
print(positive)

negative = mean_exp - standard_deviation_exp
print(negative)

#68.26% of data should be between 2.1803292952460263 and 2.8545234069979717


# In[44]:


def Count(list1, l, r):
    distribution_fraction = []
    for i in list1:
        if i > l and i < r:
            distribution_fraction.append(i)
    return len(distribution_fraction)
#return len(list(x for x in list1 if l < x < r))



list1 = a
l = 2.1803292952460263
r = 2.8545234069979717
print(Count(list1, l, r))


# In[45]:


(6830/10000)*100

#68.3% of data within +- one stndard distribution of mean
#Expected value is 68.26%
#Agrees with theory


# In[58]:


#Fraction of Distribution of Theoretical Data

positive_theor = mean_exp + standard_deviation_theor
print(positive_theor)

negative_theor = mean_exp - standard_deviation_theor
print(negative_theor)


# In[ ]:




