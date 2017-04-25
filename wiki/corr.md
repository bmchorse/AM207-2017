---
title: Correlations
shorttitle: corr
notebook: corr.ipynb
noline: 1
summary: ""
layout: wiki
---
{% assign links = site.data.wikilinks %}



```python
%matplotlib inline
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("poster")
import pymc3 as pm
```




## A gaussian with correlations

We wish to sample a 2D Posterior which looks something like below. Here the x and y axes are parameters.



```python
cov=np.array([[1,0.8],[0.8,1]])
data = np.random.multivariate_normal([0,0], cov, size=1000)
sns.kdeplot(data);
plt.scatter(data[:,0], data[:,1], alpha=0.4)
plt.xlim
```





    <function matplotlib.pyplot.xlim>




![png](corr_files/corr_3_1.png)




```python
#from https://stats.stackexchange.com/questions/260764/bayesian-correlation-matrix-estimation-with-heteroscedastic-uncertainties


```




```python
import theano.tensor as tt
def pm_make_cov(sigpriors, corr_coeffs, ndim):
    sigma_matrix = tt.nlinalg.diag(sigpriors)
    n_elem = int(ndim * (ndim - 1) / 2)
    tri_index = np.zeros([ndim, ndim], dtype=int)
    tri_index[np.triu_indices(ndim, k=1)] = np.arange(n_elem)
    tri_index[np.triu_indices(ndim, k=1)[::-1]] = np.arange(n_elem)
    corr_matrix = corr_coeffs[tri_index]
    corr_matrix = tt.fill_diagonal(corr_matrix, 1)
    return tt.nlinalg.matrix_dot(sigma_matrix, corr_matrix, sigma_matrix)
```




```python
sigs=np.array([1,1])

```




```python
tri_index = np.zeros([2, 2], dtype=int)
tri_index
```





    array([[0, 0],
           [0, 0]])





```python
with pm.Model() as modelmvg: 
    nu = pm.Uniform('nu', 1, 5)  # prior on how much correlation (0 = uniform prior on correlation, oo = no correlation)
    ndim=2
    corr_coeffs = pm.LKJCorr('corr_coeffs', nu, ndim) 
    cov = pm_make_cov(sigs, corr_coeffs)
    mvg = pm.MvNormal('mvg', mu=[0,0], cov=cov, shape=2, observed=data)
```




```python
advifit2 = pm.variational.advi( model=modelmvg, n=100000)
```


    Average ELBO = -2,487.9:  13%|█▎        | 13148/100000 [00:05<00:35, 2466.39it/s]4, 2228.80it/s]




```python
mus2, sds2, elbo2 = advifit2
mus2
```





    {'corr_coeffs_interval_': array([ 2.17362892]),
     'nu_interval_': array(-1.5325777474832887)}





```python
from pymc3.distributions.transforms import Interval
I=Interval(-1,1)
I.backward(mus2['corr_coeffs_interval_']).eval()
```





    array([ 0.79571251])





```python
with modelmvg:
    nutstrace = pm.sample(10000)
```


      7%|▋         | 14744/200000 [00:06<01:09, 2658.99it/s] | 250/200000 [00:00<01:20, 2493.47it/s]
    100%|██████████| 10000/10000 [00:35<00:00, 285.03it/s]




```python
pm.traceplot(nutstrace);
```



![png](corr_files/corr_13_0.png)




```python
pm.autocorrplot(nutstrace);
```



![png](corr_files/corr_14_0.png)




```python
pm.plot_posterior(nutstrace);
```



![png](corr_files/corr_15_0.png)




```python
with pm.Model() as modelmvg2: 
    nu = pm.Uniform('nu', 1, 5)  # prior on how much correlation (0 = uniform prior on correlation, oo = no correlation)
    ndim=2
    sigs = pm.Lognormal('sigma', np.zeros(2), np.ones(2), shape=2)
    corr_coeffs = pm.LKJCorr('corr_coeffs', nu, ndim) 
    cov = pm_make_cov(sigs, corr_coeffs, ndim)
    mvg = pm.MvNormal('mvg', mu=[0,0], cov=cov, shape=2, observed=data)
```




```python
with modelmvg2:
    nutstrace2 = pm.sample(10000)
```


    Average ELBO = -2,887.7:  11%|█         | 21663/200000 [00:09<01:16, 2342.61it/s]8, 1687.99it/s]
    100%|██████████| 10000/10000 [00:42<00:00, 233.41it/s]




```python
pm.plot_posterior(nutstrace2);
```



![png](corr_files/corr_18_0.png)

