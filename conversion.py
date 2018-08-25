import pymc3 as pm
import numpy as np
import theano.tensor as tt

variant_a = [np.random.randint(28,32) for varinat_a in range(120)] #120 Samples for variant A - THIS IS CONTROL/DEFAULT
variant_b = [np.random.randint(32,34) for varinat_b in range(120)] #120 Samples for variant B
variant_c = [np.random.randint(30,36) for varinat_b in range(120)] #dummy
uplift = 1.1 # representing 10%
with pm.Model() as model:
    alpha_a = 1.0/np.mean(variant_a)
    alpha_b = 1.0/np.mean(variant_b)
    alpha_c = 1.0/np.mean(variant_c)
    lambda_a = pm.Exponential("lambda_a", alpha_a)
    lambda_b = pm.Exponential("lambda_b", alpha_b)
    lambda_c = pm.Exponential("lambda_c", alpha_c)
    delta_ba = pm.Deterministic("delta_ba", lambda_b - lambda_a)
    delta_ca = pm.Deterministic("delta_ca", lambda_c - lambda_a) #marginal
    delta_ba_uplift = pm.Deterministic("delta_ba_uplift", lambda_b - uplift*lambda_a) #for 10%
    delta_ca_uplift = pm.Deterministic("delta_ca_uplift", lambda_c - uplift*lambda_a)

with model:
    observation_a = pm.Poisson("obs_a", lambda_a, observed=variant_a)
    observation_b = pm.Poisson("obs_b", lambda_b, observed=variant_b)
    observation_c = pm.Poisson("obs_c", lambda_c, observed=variant_c)

with model:
    step = pm.Metropolis()
    trace = pm.sample(10000, tune=5000,step=step)

#Variant A
lambda_a_samples = trace['lambda_a']
lambda_a_samples = lambda_a_samples[1000:] # burned trace
#Variant B
lambda_b_samples = trace['lambda_b']
lambda_b_samples = lambda_b_samples[1000:] # burned trace
#Variant C
lambda_c_samples = trace['lambda_c']
lambda_c_samples = lambda_c_samples[1000:] # burned trace

#Variant B - Variant A
delta_samples_ba = trace['delta_ba']
delta_samples_ba_uplift = trace['delta_ba_uplift']
delta_samples_ba = delta_samples_ba[1000:] #burned trace
delta_samples_ba_uplift = delta_samples_ba_uplift[1000:]
#Variant C - Variant A
delta_samples_ca = trace['delta_ca']
delta_samples_ca_uplift = trace['delta_ca_uplift']
delta_samples_ca = delta_samples_ca[1000:] #burned trace
delta_samples_ca_uplift = delta_samples_ca_uplift[1000:]

#range of variant_a
print np.mean(lambda_a_samples) #Conversion
print np.min(lambda_a_samples) #Min 
print np.max(lambda_a_samples) #max

#range of variant_b
print np.mean(lambda_b_samples)
print np.min(lambda_b_samples)
print np.max(lambda_b_samples)

#range of variant_c
print np.mean(lambda_c_samples)
print np.min(lambda_c_samples)
print np.max(lambda_c_samples)


print (np.mean(delta_samples_ba>0)) #probabilility that Variant B will win marginally
print (np.mean(delta_samples_ca>0)) #probabilility that Variant B will win marginally
print (np.mean(delta_samples_ba_uplift > 0)) #probabilility that Variant B will win by uplift of 10%
print (np.mean(delta_samples_ca_uplift > 0)) #probabilility that Variant B will win by uplift of 10%
