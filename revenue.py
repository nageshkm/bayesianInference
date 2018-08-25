import pymc3 as pm
import numpy as np
import theano.tensor as tt

variant_a = [np.random.randint(28,32) for varinat_a in range(120)] #120 Samples for variant A - THIS IS CONTROL/DEFAULT
variant_b = [np.random.randint(32,34) for varinat_b in range(120)] #120 Samples for variant B
variant_c = [np.random.randint(30,36) for varinat_b in range(120)] #dummy
uplift = 1.1 #10% uplift
with pm.Model() as model:
    
    # modelling an unknown standard deviation for the variants - Uniform
    sd_a = pm.Uniform("sd_a", 0, 5*np.std(variant_a, axis=0))
    sd_b = pm.Uniform("sd_b", 0, 5*np.std(variant_b, axis=0))
    sd_c = pm.Uniform("sd_c", 0, 5*np.std(variant_b, axis=0))

    # modelling an unknown mean for the variants - Gaussian
    mean_a = pm.Normal("mean_a", mu=np.mean(variant_a), sd=sd_a)
    mean_b = pm.Normal("mean_b", mu=np.mean(variant_b), sd=sd_b)
    mean_c = pm.Normal("mean_c", mu=np.mean(variant_c), sd=sd_c)

    # correlating the model with the actual data
    observation_a = pm.Normal("obs_a", mu=mean_a, sd=sd_a, observed=variant_a)
    observation_b = pm.Normal("obs_b", mu=mean_b, sd=sd_b, observed=variant_b)
    observation_c = pm.Normal("obs_c", mu=mean_b, sd=sd_c, observed=variant_c)
    # measuring uplift
    delta_ba = pm.Deterministic("delta_ba", mean_b - mean_a) #marginal VarB - Control
    delta_ca = pm.Deterministic("delta_ca", mean_c - mean_a) #marginal VarC - Control
    delta_ba_uplift = pm.Deterministic("delta_ba_uplift", mean_b - uplift*mean_a) #10% uplift VarB - Control
    delta_ca_uplift = pm.Deterministic("delta_ca_uplift", mean_c - uplift*mean_a) #10% uplift VarC - Control

with model:
    step = pm.Metropolis()
    trace = pm.sample(10000, tune=5000,step=step)

#Indicates the mean and std_dev of variant a
var_a_samples = trace['mean_a']
var_a_samples = var_a_samples[1000:] #burned trace
sd_a_samples = trace['sd_a']

#Indicates the mean and std_dev of variant b
var_b_samples = trace['mean_b']
var_b_samples = var_b_samples[1000:] #burned trace
sd_b_samples = trace['sd_b']

#Indicates the mean and std_dev of variant c
var_c_samples = trace['mean_c']
var_c_samples = var_c_samples[1000:] #burned trace
sd_c_samples = trace['sd_c']

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
print np.mean(var_a_samples) #Conversion
print (np.min(var_a_samples) - 3*np.max(sd_a_samples)) #Min - 3*SD
print (np.max(var_a_samples) + 3*np.max(sd_a_samples)) #max + 3*SD

#range of variant_b
print np.mean(var_b_samples)
print (np.min(var_b_samples) - 3*np.max(sd_b_samples)) #Min - 3*SD
print (np.max(var_b_samples) + 3*np.max(sd_b_samples)) #max + 3*SD

#range of variant_c
print np.mean(var_c_samples)
print (np.min(var_c_samples) - 3*np.max(sd_c_samples)) #Min - 3*SD
print (np.max(var_c_samples) + 3*np.max(sd_c_samples)) #max + 3*SD


print (np.mean(delta_samples_ba>0)) #probabilility that Variant B will win marginally
print (np.mean(delta_samples_ca>0)) #probabilility that Variant B will win marginally
print (np.mean(delta_samples_ba_uplift > 0)) #probabilility that Variant B will win by uplift of 10%
print (np.mean(delta_samples_ca_uplift > 0)) #probabilility that Variant B will win by uplift of 10%
