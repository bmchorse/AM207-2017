---
title: Gaussian Mixture Model with ADVI
shorttitle: gaussian-mixture-model-advi
notebook: gaussian-mixture-model-advi.ipynb
noline: 1
summary: ""
layout: wiki
---
{% assign links = site.data.wikilinks %}

This notebook is taken almost verbatim from the pymc3 documentation. Its a very good illustration of what needs to be done to

- marginalize over discretes for ADVI
- deal with 2D mixtures
- get MCMC done as well with a custom density



```python
%matplotlib inline

import theano
theano.config.floatX = 'float64'

import pymc3 as pm
from pymc3 import Normal, Metropolis, sample, MvNormal, Dirichlet, \
    DensityDist, find_MAP, NUTS, Slice
import theano.tensor as tt
from theano.tensor.nlinalg import det
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("poster")
```


## Generate some data



```python
n_samples = 100
# we can use this to set up the pseodorandom number appropriately to communicate results reliably
rng = np.random.RandomState(123)
ms = np.array([[-1, -1.5], [1, 1]])
ps = np.array([0.2, 0.8])

zs = np.array([rng.multinomial(1, ps) for _ in range(n_samples)]).T
xs = [z[:, np.newaxis] * rng.multivariate_normal(m, np.eye(2), size=n_samples)
      for z, m in zip(zs, ms)]
data = np.sum(np.dstack(xs), axis=2)

plt.scatter(data[:, 0], data[:, 1], c='g', alpha=0.5)
plt.scatter(ms[0, 0], ms[0, 1], c='r', s=100)
plt.scatter(ms[1, 0], ms[1, 1], c='b', s=100)
```





    <matplotlib.collections.PathCollection at 0x122305cc0>




![png](gaussian-mixture-model-advi_files/gaussian-mixture-model-advi_3_1.png)




```python
zs[0,:].mean(), zs[1,:].mean()
```





    (0.13, 0.87)



## Marginalize discretes

Gaussian mixture models are usually constructed with categorical random variables. However, any discrete rvs does not fit ADVI. Here, class assignment variables are marginalized out, giving weighted sum of the probability for the gaussian components. The log likelihood of the total probability is calculated using logsumexp, which is a standard technique for making this kind of calculation stable. 

In the below code, DensityDist class is used as the likelihood term. The second argument, logp_gmix(mus, pi, np.eye(2)), is a python function which recieves observations (denoted by 'value') and returns the tensor representation of the log-likelihood. 



```python
from pymc3.math import logsumexp

# Log likelihood of normal distribution
def logp_normal(mu, tau, value):
    # log probability of individual samples
    k = tau.shape[0]
    delta = lambda mu: value - mu
    return (-1 / 2.) * (k * tt.log(2 * np.pi) + tt.log(1./det(tau)) +
                         (delta(mu).dot(tau) * delta(mu)).sum(axis=1))

# Log likelihood of Gaussian mixture distribution
def logp_gmix(mus, pi, tau):
    def logp_(value):        
        logps = [tt.log(pi[i]) + logp_normal(mu, tau, value)
                 for i, mu in enumerate(mus)]
            
        return tt.sum(logsumexp(tt.stacklists(logps)[:, :n_samples], axis=0))

    return logp_
```


## Sampling



```python
with pm.Model() as model:
    mus = [MvNormal('mu_%d' % i, mu=np.zeros(2), tau=np.eye(2), shape=(2,))
           for i in range(2)]
    pi = Dirichlet('pi', a=0.1 * np.ones(2), shape=(2,))
    xs = DensityDist('x', logp_gmix(mus, pi, np.eye(2)), observed=data)
```


For comparison with ADVI, run MCMC. 



```python
with model:
    start = find_MAP()
    step = Metropolis()
    trace = sample(1000, step, start=start)
```


    Optimization terminated successfully.
             Current function value: 343.215633
             Iterations: 2
             Function evaluations: 3
             Gradient evaluations: 3


    100%|██████████| 1000/1000 [00:01<00:00, 877.03it/s]   | 1/1000 [00:00<02:26,  6.83it/s]


Check posterior of component means and weights. We can see that the MCMC samples of the component mean for the lower-left component varied more than the upper-right due to the difference of the sample size of these clusters. 



```python
plt.scatter(data[:, 0], data[:, 1], alpha=0.5, c='g')
mu_0, mu_1 = trace['mu_0'], trace['mu_1']
plt.scatter(mu_0[-500:, 0], mu_0[-500:, 1], c="r", s=10, alpha=0.1)
plt.scatter(mu_1[-500:, 0], mu_1[-500:, 1], c="b", s=10, alpha=0.1)
plt.xlim(-6, 6)
plt.ylim(-6, 6)
```





    (-6, 6)




![png](gaussian-mixture-model-advi_files/gaussian-mixture-model-advi_13_1.png)




```python
sns.barplot([1, 2], np.mean(trace['pi'][-500:], axis=0), 
            palette=['red', 'blue'])
```





    <matplotlib.axes._subplots.AxesSubplot at 0x11f291630>




![png](gaussian-mixture-model-advi_files/gaussian-mixture-model-advi_14_1.png)


We can use the same model with ADVI as follows. 



```python
with pm.Model() as model:
    mus = [MvNormal('mu_%d' % i, mu=np.zeros(2), tau=np.eye(2), shape=(2,))
           for i in range(2)]
    pi = Dirichlet('pi', a=0.1 * np.ones(2), shape=(2,))
    xs = DensityDist('x', logp_gmix(mus, pi, np.eye(2)), observed=data)
    
%time means, sds, elbos = pm.variational.advi( \
    model=model, n=1000, learning_rate=1e-1)
```


      0%|          | 0/1000 [00:00<?, ?it/s]Average ELBO = -321.81:  38%|███▊      | 383/1000 [00:00<00:00, 3820.49it/s]

    CPU times: user 3.29 s, sys: 64.3 ms, total: 3.35 s
    Wall time: 3.71 s


    


The function returns three variables. 'means' and 'sds' are the mean and standart deviations of the variational posterior. Note that these values are in the transformed space, not in the original space. For random variables in the real line, e.g., means of the Gaussian components, no transformation is applied. Then we can see the variational posterior in the original space. 



```python
from copy import deepcopy

mu_0, sd_0 = means['mu_0'], sds['mu_0']
mu_1, sd_1 = means['mu_1'], sds['mu_1']

def logp_normal_np(mu, tau, value):
    # log probability of individual samples
    k = tau.shape[0]
    delta = lambda mu: value - mu
    return (-1 / 2.) * (k * np.log(2 * np.pi) + np.log(1./np.linalg.det(tau)) +
                         (delta(mu).dot(tau) * delta(mu)).sum(axis=1))

def threshold(zz):    
    zz_ = deepcopy(zz)
    zz_[zz < np.max(zz) * 1e-2] = None
    return zz_

def plot_logp_normal(ax, mu, sd, cmap):
    f = lambda value: np.exp(logp_normal_np(mu, np.diag(1 / sd**2), value))
    g = lambda mu, sd: np.arange(mu - 3, mu + 3, .1)
    xx, yy = np.meshgrid(g(mu[0], sd[0]), g(mu[1], sd[1]))
    zz = f(np.vstack((xx.reshape(-1), yy.reshape(-1))).T).reshape(xx.shape)
    ax.contourf(xx, yy, threshold(zz), cmap=cmap, alpha=0.9)
           
fig, ax = plt.subplots()
plt.scatter(data[:, 0], data[:, 1], alpha=0.5, c='g')
plot_logp_normal(ax, mu_0, sd_0, cmap='Reds')
plot_logp_normal(ax, mu_1, sd_1, cmap='Blues')
plt.xlim(-6, 6)
plt.ylim(-6, 6)
```





    (-6, 6)




![png](gaussian-mixture-model-advi_files/gaussian-mixture-model-advi_18_1.png)




```python
model.vars, means
```





    ([mu_0, mu_1, pi_stickbreaking_],
     {'mu_0': array([ 0.80874025,  0.97874877]),
      'mu_1': array([-0.70773468, -1.49193359]),
      'pi_stickbreaking_': array([ 1.99972356])})



We need to backward-transform 'pi', which is transformed by 'stick_breaking'. Variables that are transformed are not transformed back by the current pymc3 implementation. This is a gotcha to keep in mind. See https://github.com/pymc-devs/pymc3/blob/master/pymc3/distributions/transforms.py#L200 for details



```python
model.pi_stickbreaking_
```





    pi_stickbreaking_





```python
from pymc3.distributions.transforms import StickBreaking
trans = StickBreaking()
trans.backward(means['pi_stickbreaking_']).eval()
```





    array([ 0.88076805,  0.11923195])



'elbos' contains the trace of ELBO, showing stochastic convergence of the algorithm. 



```python
plt.plot(elbos)
```





    [<matplotlib.lines.Line2D at 0x11d7f1470>]




![png](gaussian-mixture-model-advi_files/gaussian-mixture-model-advi_24_1.png)


To demonstrate that ADVI works for large dataset with mini-batch, let's create 100,000 samples from the same mixture distribution. 



```python
n_samples = 100000

zs = np.array([rng.multinomial(1, ps) for _ in range(n_samples)]).T
xs = [z[:, np.newaxis] * rng.multivariate_normal(m, np.eye(2), size=n_samples)
      for z, m in zip(zs, ms)]
data = np.sum(np.dstack(xs), axis=2)

plt.scatter(data[:, 0], data[:, 1], c='g', alpha=0.008)
plt.scatter(ms[0, 0], ms[0, 1], c='r', s=100)
plt.scatter(ms[1, 0], ms[1, 1], c='b', s=100)
plt.xlim(-6, 6)
plt.ylim(-6, 6)
```





    (-6, 6)




![png](gaussian-mixture-model-advi_files/gaussian-mixture-model-advi_26_1.png)


MCMC takes of ther order of a minute in time, which is 50 times more than on the small dataset.



```python
with pm.Model() as model:
    mus = [MvNormal('mu_%d' % i, mu=np.zeros(2), tau=np.eye(2), shape=(2,))
           for i in range(2)]
    pi = Dirichlet('pi', a=0.1 * np.ones(2), shape=(2,))
    xs = DensityDist('x', logp_gmix(mus, pi, np.eye(2)), observed=data)
    
    start = find_MAP()
    step = Metropolis()
    trace = sample(1000, step, start=start)
```


    Optimization terminated successfully.
             Current function value: 365851.269817
             Iterations: 2
             Function evaluations: 3
             Gradient evaluations: 3


    100%|██████████| 1000/1000 [01:06<00:00, 15.14it/s]    | 1/1000 [00:00<03:24,  4.89it/s]


Posterior samples are concentrated on the true means, so looks like single point for each component. 



```python
plt.scatter(data[:, 0], data[:, 1], alpha=0.001, c='g')
mu_0, mu_1 = trace['mu_0'], trace['mu_1']
plt.scatter(mu_0[-500:, 0], mu_0[-500:, 1], c="r", s=50)
plt.scatter(mu_1[-500:, 0], mu_1[-500:, 1], c="b", s=50)
plt.xlim(-6, 6)
plt.ylim(-6, 6)
```





    (-6, 6)




![png](gaussian-mixture-model-advi_files/gaussian-mixture-model-advi_30_1.png)


For ADVI with mini-batch, put theano tensor on the observed variable of the ObservedRV. The tensor will be replaced with mini-batches. Because of the difference of the size of mini-batch and whole samples, the log-likelihood term should be appropriately scaled. To tell the log-likelihood term, we need to give ObservedRV objects ('minibatch_RVs' below) where mini-batch is put. Also we should keep the tensor ('minibatch_tensors'). 



```python
data_t = tt.matrix()
data_t.tag.test_value = np.zeros((1, 2)).astype(float)

with pm.Model() as model:
    mus = [MvNormal('mu_%d' % i, mu=np.zeros(2), tau=0.1 * np.eye(2), shape=(2,))
           for i in range(2)]
    pi = Dirichlet('pi', a=0.1 * np.ones(2), shape=(2,))
    xs = DensityDist('x', logp_gmix(mus, pi, np.eye(2)), observed=data_t)
    
minibatch_tensors = [data_t]
minibatch_RVs = [xs]
```


Make a generator for mini-batches of size 200. Here, we take random sampling strategy to make mini-batches. 



```python
def create_minibatch(data):
    rng = np.random.RandomState(0)
    
    while True:
        ixs = rng.randint(len(data), size=200)
        yield [data[ixs]]

minibatches = create_minibatch(data)
total_size = len(data)
```


Run ADVI. It's much faster than MCMC, though the problem here is simple and it's not a fair comparison. 



```python
# Used only to write the function call in single line for using %time
# is there more smart way?
def f():
    return pm.variational.advi_minibatch(
    model=model, n=1000, minibatch_tensors=minibatch_tensors, 
    minibatch_RVs=minibatch_RVs, minibatches=minibatches,
    total_size=total_size, learning_rate=1e-1)

%time means, sds, elbos = f()
```


    Average ELBO = -322,294.13: 100%|██████████| 1000/1000 [00:01<00:00, 867.48it/s]58.24it/s]

    CPU times: user 13.8 s, sys: 1.14 s, total: 14.9 s
    Wall time: 16.2 s


    


The result is almost the same. 



```python
from copy import deepcopy

mu_0, sd_0 = means['mu_0'], sds['mu_0']
mu_1, sd_1 = means['mu_1'], sds['mu_1']

fig, ax = plt.subplots()
plt.scatter(data[:, 0], data[:, 1], alpha=0.001, c='g')
plt.scatter(mu_0[0], mu_0[1], c="r", s=50)
plt.scatter(mu_1[0], mu_1[1], c="b", s=50)
plt.xlim(-6, 6)
plt.ylim(-6, 6)
```





    (-6, 6)




![png](gaussian-mixture-model-advi_files/gaussian-mixture-model-advi_38_1.png)


The variance of the trace of ELBO is larger than without mini-batch because of the subsampling from the whole samples. 



```python
plt.plot(elbos);
```



![png](gaussian-mixture-model-advi_files/gaussian-mixture-model-advi_40_0.png)

