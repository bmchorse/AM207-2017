---
title: Poisson Regression - tools on islands part 2
shorttitle: Islands2
notebook: Islands2.ipynb
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




We go back to our island tools data set to illustrate

- model comparison using WAIC
- model averaging using WAIC
- fighting overdispersion by making a hierarchical regression model.



```python
df=pd.read_csv("islands.csv", sep=';')
df
```





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>culture</th>
      <th>population</th>
      <th>contact</th>
      <th>total_tools</th>
      <th>mean_TU</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Malekula</td>
      <td>1100</td>
      <td>low</td>
      <td>13</td>
      <td>3.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Tikopia</td>
      <td>1500</td>
      <td>low</td>
      <td>22</td>
      <td>4.7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Santa Cruz</td>
      <td>3600</td>
      <td>low</td>
      <td>24</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Yap</td>
      <td>4791</td>
      <td>high</td>
      <td>43</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Lau Fiji</td>
      <td>7400</td>
      <td>high</td>
      <td>33</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Trobriand</td>
      <td>8000</td>
      <td>high</td>
      <td>19</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Chuuk</td>
      <td>9200</td>
      <td>high</td>
      <td>40</td>
      <td>3.8</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Manus</td>
      <td>13000</td>
      <td>low</td>
      <td>28</td>
      <td>6.6</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Tonga</td>
      <td>17500</td>
      <td>high</td>
      <td>55</td>
      <td>5.4</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Hawaii</td>
      <td>275000</td>
      <td>low</td>
      <td>71</td>
      <td>6.6</td>
    </tr>
  </tbody>
</table>
</div>





```python
df['logpop']=np.log(df.population)
df['clevel']=(df.contact=='high')*1
df
```





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>culture</th>
      <th>population</th>
      <th>contact</th>
      <th>total_tools</th>
      <th>mean_TU</th>
      <th>logpop</th>
      <th>clevel</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Malekula</td>
      <td>1100</td>
      <td>low</td>
      <td>13</td>
      <td>3.2</td>
      <td>7.003065</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Tikopia</td>
      <td>1500</td>
      <td>low</td>
      <td>22</td>
      <td>4.7</td>
      <td>7.313220</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Santa Cruz</td>
      <td>3600</td>
      <td>low</td>
      <td>24</td>
      <td>4.0</td>
      <td>8.188689</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Yap</td>
      <td>4791</td>
      <td>high</td>
      <td>43</td>
      <td>5.0</td>
      <td>8.474494</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Lau Fiji</td>
      <td>7400</td>
      <td>high</td>
      <td>33</td>
      <td>5.0</td>
      <td>8.909235</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Trobriand</td>
      <td>8000</td>
      <td>high</td>
      <td>19</td>
      <td>4.0</td>
      <td>8.987197</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Chuuk</td>
      <td>9200</td>
      <td>high</td>
      <td>40</td>
      <td>3.8</td>
      <td>9.126959</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Manus</td>
      <td>13000</td>
      <td>low</td>
      <td>28</td>
      <td>6.6</td>
      <td>9.472705</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Tonga</td>
      <td>17500</td>
      <td>high</td>
      <td>55</td>
      <td>5.4</td>
      <td>9.769956</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Hawaii</td>
      <td>275000</td>
      <td>low</td>
      <td>71</td>
      <td>6.6</td>
      <td>12.524526</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>





```python
def postscat(trace, thevars):
    d={}
    for v in thevars:
        d[v] = trace.get_values(v)
    df = pd.DataFrame.from_dict(d)
    return sns.pairplot(df, diag_kind="kde")
```


## Centered Model

As usual, centering the log-population fixes things:



```python
df.logpop_c = df.logpop - df.logpop.mean()
```




```python
from theano import tensor as t
with pm.Model() as m1c:
    betap = pm.Normal("betap", 0, 1)
    betac = pm.Normal("betac", 0, 1)
    betapc = pm.Normal("betapc", 0, 1)
    alpha = pm.Normal("alpha", 0, 100)
    loglam = alpha + betap*df.logpop_c + betac*df.clevel + betapc*df.clevel*df.logpop_c
    y = pm.Poisson("ntools", mu=t.exp(loglam), observed=df.total_tools)
```




```python
with m1c:
    trace1c = pm.sample(5000, njobs=2)
```


    Average ELBO = -51.061: 100%|██████████| 200000/200000 [00:16<00:00, 12467.18it/s]9, 10447.72it/s]
    100%|██████████| 5000/5000 [00:10<00:00, 480.61it/s]




```python
pm.traceplot(trace1c);
```



![png](Islands2_files/Islands2_10_0.png)




```python
pm.autocorrplot(trace1c);
```



![png](Islands2_files/Islands2_11_0.png)




```python
pm.effective_n(trace1c)
```





    {'alpha': 3788.0, 'betac': 3879.0, 'betap': 6771.0, 'betapc': 10000.0}





```python
postscat(trace1c,trace1c.varnames);
```


    //anaconda/envs/py35/lib/python3.5/site-packages/statsmodels/nonparametric/kdetools.py:20: VisibleDeprecationWarning: using a non-integer number instead of an integer will result in an error in the future
      y = X[:m/2+1] + np.r_[0,X[m/2+1:],0]*1j



![png](Islands2_files/Islands2_13_1.png)




```python
pm.plot_posterior(trace1c);
```



![png](Islands2_files/Islands2_14_0.png)


## Model comparison for interaction significance

This is an example of feature selection, where we want to decide whether we should keep the interaction term or not, that is, whether the interaction is significant or not? We'll use model comparison to achieve this!

We can see some summary stats from this model:



```python
dfsum=pm.df_summary(trace1c)
dfsum
```





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>sd</th>
      <th>mc_error</th>
      <th>hpd_2.5</th>
      <th>hpd_97.5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>betap</th>
      <td>0.262858</td>
      <td>0.035277</td>
      <td>0.000366</td>
      <td>0.193673</td>
      <td>0.331655</td>
    </tr>
    <tr>
      <th>betac</th>
      <td>0.284060</td>
      <td>0.117721</td>
      <td>0.001697</td>
      <td>0.061976</td>
      <td>0.520949</td>
    </tr>
    <tr>
      <th>betapc</th>
      <td>0.066318</td>
      <td>0.169238</td>
      <td>0.001547</td>
      <td>-0.251977</td>
      <td>0.406100</td>
    </tr>
    <tr>
      <th>alpha</th>
      <td>3.312385</td>
      <td>0.089722</td>
      <td>0.001343</td>
      <td>3.142028</td>
      <td>3.490543</td>
    </tr>
  </tbody>
</table>
</div>





```python
pm.dic(trace1c, m1c)
```





    95.58548599529469





```python
pm.waic(trace1c, m1c)
```


            log predictive densities exceeds 0.4. This could be indication of
            WAIC starting to fail see http://arxiv.org/abs/1507.04544 for details
            
      """)





    WAIC_r(WAIC=83.861714942970877, WAIC_se=12.202689681783035, p_WAIC=6.9477629505774319)



### Sampling from multiple different centered models

**(A)** Our complete model

**(B)** A model with no interaction



```python
with pm.Model() as m2c_nopc:
    betap = pm.Normal("betap", 0, 1)
    betac = pm.Normal("betac", 0, 1)
    alpha = pm.Normal("alpha", 0, 100)
    loglam = alpha + betap*df.logpop_c + betac*df.clevel
    y = pm.Poisson("ntools", mu=t.exp(loglam), observed=df.total_tools)
    trace2c_nopc = pm.sample(5000, njobs=2)
```


    Average ELBO = -49.275: 100%|██████████| 200000/200000 [00:15<00:00, 12971.12it/s]7, 11249.31it/s]
    100%|██████████| 5000/5000 [00:09<00:00, 539.61it/s]


**(C)** A model with no contact term



```python
with pm.Model() as m2c_onlyp:
    betap = pm.Normal("betap", 0, 1)
    alpha = pm.Normal("alpha", 0, 100)
    loglam = alpha + betap*df.logpop_c
    y = pm.Poisson("ntools", mu=t.exp(loglam), observed=df.total_tools)
    trace2c_onlyp = pm.sample(5000, njobs=2)
```


    Average ELBO = -50.136: 100%|██████████| 200000/200000 [00:12<00:00, 15434.86it/s]6, 12351.12it/s]
    100%|██████████| 5000/5000 [00:05<00:00, 902.48it/s] 


**(D)** A model with only the contact term



```python
with pm.Model() as m2c_onlyc:
    betac = pm.Normal("betac", 0, 1)
    alpha = pm.Normal("alpha", 0, 100)
    loglam = alpha +  betac*df.clevel
    y = pm.Poisson("ntools", mu=t.exp(loglam), observed=df.total_tools)
    trace2c_onlyc = pm.sample(5000, njobs=2)
```


    Average ELBO = -74.472:  27%|██▋       | 53752/200000 [00:03<00:09, 14980.29it/s]16, 11808.13it/s]
    100%|██████████| 5000/5000 [00:07<00:00, 684.44it/s]


**(E)** A model with only the intercept.



```python
with pm.Model() as m2c_onlyic:
    alpha = pm.Normal("alpha", 0, 100)
    loglam = alpha
    y = pm.Poisson("ntools", mu=t.exp(loglam), observed=df.total_tools)
    trace2c_onlyic = pm.sample(5000, njobs=2)
```


    Average ELBO = -297.49:  15%|█▌        | 30824/200000 [00:02<00:11, 15294.24it/s]14, 13683.81it/s]
    100%|██████████| 5000/5000 [00:03<00:00, 1272.31it/s]


We create a dictionary from these models and their traces, so that we can track the names as well



```python
modeldict=dict(m1c=(m1c, trace1c), m2c_nopc = (m2c_nopc, trace2c_nopc),
              m2c_onlyp=(m2c_onlyp, trace2c_onlyp),
              m2c_onlyc=(m2c_onlyc, trace2c_onlyc),
              m2c_onlyic=(m2c_onlyic, trace2c_onlyic))
```




```python
names, models, traces=zip(*[(a, b, c) for a, (b, c) in modeldict.items()])
```


## Comparing the models using WAIC

Finally we use `pm.compare` to create a dataframe of comparisions, and do some pandas stuff to label the rows.



```python
comparedf = pm.compare(traces, models)
temp=comparedf.sort_index()
temp['name']=names
comparedf = temp.sort('WAIC').set_index('name')
comparedf
```


    //anaconda/envs/py35/lib/python3.5/site-packages/ipykernel/__main__.py:4: FutureWarning: sort(columns=....) is deprecated, use sort_values(by=.....)





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>WAIC</th>
      <th>pWAIC</th>
      <th>dWAIC</th>
      <th>weight</th>
      <th>SE</th>
      <th>dSE</th>
      <th>warning</th>
    </tr>
    <tr>
      <th>name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>m2c_nopc</th>
      <td>79.3591</td>
      <td>4.39013</td>
      <td>0</td>
      <td>0.846327</td>
      <td>11.0543</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>m1c</th>
      <td>83.8617</td>
      <td>6.94776</td>
      <td>4.50259</td>
      <td>0.0890866</td>
      <td>12.2027</td>
      <td>4.00079</td>
      <td>1</td>
    </tr>
    <tr>
      <th>m2c_onlyp</th>
      <td>84.5049</td>
      <td>3.77558</td>
      <td>5.14581</td>
      <td>0.0645862</td>
      <td>8.91335</td>
      <td>19.3282</td>
      <td>1</td>
    </tr>
    <tr>
      <th>m2c_onlyic</th>
      <td>141.327</td>
      <td>8.10745</td>
      <td>61.9681</td>
      <td>2.96038e-14</td>
      <td>31.6664</td>
      <td>339.158</td>
      <td>1</td>
    </tr>
    <tr>
      <th>m2c_onlyc</th>
      <td>152.975</td>
      <td>18.1559</td>
      <td>73.6157</td>
      <td>8.75158e-17</td>
      <td>46.6488</td>
      <td>679.109</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



From McElreath, here is how to read this table:

>(1)	WAIC is obviously WAIC for each model. Smaller WAIC indicates better estimated out-of-sample deviance.

>(2)	pWAIC is the estimated effective number of parameters. This provides a clue as to how flexible each model is in fitting the sample.

>(3)	dWAIC is the difference between each WAIC and the lowest WAIC. Since only relative deviance matters, this column shows the differences in relative fashion.

>(4)	weight is the AKAIKE WEIGHT for each model. These values are transformed information criterion values. I'll explain them below.

>(5)	SE is the standard error of the WAIC estimate. WAIC is an estimate, and provided the sample size N is large enough, its uncertainty will be well approximated by its standard error. So this SE value isn't necessarily very precise, but it does provide a check against overconfidence in differences between WAIC values.

>(6)	dSE is the standard error of the difference in WAIC between each model and the top-ranked model. So it is missing for the top model. 

>The weight for a model i in a set of m models is given by:

$$w_i = \frac{exp(-\frac{1}{2}dWAIC_i)}{\sum_j exp(-\frac{1}{2}dWAIC_j)}$$

>The Akaike weight formula might look rather odd, but really all it is doing is putting WAIC on a probability scale, so it just undoes the multiplication by −2 and then exponentiates to reverse the log transformation. Then it standardizes by dividing by the total. So each weight will be a number from 0 to 1, and the weights together always sum to 1. Now larger values are better.

>But what do these weights mean? 

>Akaike's interpretation:

>A model's weight is an estimate of the probability that the model will make the best predictions on new data, conditional on the set of models considered...the Akaike weights are analogous to posterior probabilities of models, conditional on expected future data.

>So you can heuristically read each weight as an estimated probability that each model will perform best on future data. In simulation at least, interpreting weights in this way turns out to be appropriate. (McElreath 199-200)



```python
def compare_plot(comp_df, ax=None):
    """
    Model comparison summary plot in the style of the one used in the book
    Statistical Rethinking by Richard McElreath.
    Parameters
    ----------
    comp_df: DataFrame
        The result of the pm.compare() function
    ax : axes
        Matplotlib axes. Defaults to None.
    Returns
    -------
    ax : matplotlib axes
    """
    if ax is None:
        _, ax = plt.subplots()

    yticks_pos, step = np.linspace(0, -1, (comp_df.shape[0] * 2) - 1, retstep=True)
    yticks_pos[1::2] = yticks_pos[1::2] + step / 2

    yticks_labels = [''] * len(yticks_pos)
    yticks_labels[0] = comp_df.index[0]
    yticks_labels[1::2] = comp_df.index[1:]

    data = comp_df.values
    min_ic = data[0, 0]

    ax.errorbar(x=data[:, 0], y=yticks_pos[::2], xerr=data[:, 4],
                fmt='ko', mfc='None', mew=1)
    ax.errorbar(x=data[1:, 0], y=yticks_pos[1::2],
                xerr=data[1:, 5], fmt='^', color='grey')

    ax.plot(data[:, 0] - (2 * data[:, 1]), yticks_pos[::2], 'ko')
    ax.axvline(min_ic, ls='--', color='grey')

    ax.set_yticks(yticks_pos)
    ax.set_yticklabels(yticks_labels)
    ax.set_xlabel('Deviance')
    lims=ax.get_ylim()
    ax.set_ylim(lims[0] + step/2., lims[1] - step/2.)
    return ax
```


We can make visual comparison plots in the style of McElreath's book. We can see that all the weight is in the no-interaction, full, and only log(population) models.



```python
compare_plot(comparedf)
```





    <matplotlib.axes._subplots.AxesSubplot at 0x12686fcc0>




![png](Islands2_files/Islands2_35_1.png)


### Comparing for non-centered models

We can redo the coparison for non-centered models



```python
with pm.Model() as m1:
    betap = pm.Normal("betap", 0, 1)
    betac = pm.Normal("betac", 0, 1)
    betapc = pm.Normal("betapc", 0, 1)
    alpha = pm.Normal("alpha", 0, 100)
    loglam = alpha + betap*df.logpop + betac*df.clevel + betapc*df.clevel*df.logpop
    y = pm.Poisson("ntools", mu=t.exp(loglam), observed=df.total_tools)
    trace1 = pm.sample(5000, njobs=2)
```


    Average ELBO = -55.784: 100%|██████████| 200000/200000 [00:19<00:00, 10524.14it/s]   12869.09it/s]
    100%|██████████| 5000/5000 [00:55<00:00, 90.06it/s] 




```python
with pm.Model() as m2_onlyp:
    betap = pm.Normal("betap", 0, 1)
    alpha = pm.Normal("alpha", 0, 100)
    loglam = alpha + betap*df.logpop
    y = pm.Poisson("ntools", mu=t.exp(loglam), observed=df.total_tools)
    trace2_onlyp = pm.sample(5000, njobs=2)
```


    Average ELBO = -51.832: 100%|██████████| 200000/200000 [00:18<00:00, 10680.42it/s]   12642.66it/s]
    100%|██████████| 5000/5000 [00:21<00:00, 233.85it/s]




```python
with pm.Model() as m2_nopc:
    betap = pm.Normal("betap", 0, 1)
    betac = pm.Normal("betac", 0, 1)
    alpha = pm.Normal("alpha", 0, 100)
    loglam = alpha + betap*df.logpop + betac*df.clevel
    y = pm.Poisson("ntools", mu=t.exp(loglam), observed=df.total_tools)
    trace2_nopc = pm.sample(5000, njobs=2)
```


    Average ELBO = -51.024: 100%|██████████| 200000/200000 [00:18<00:00, 10974.96it/s]   188.35it/s]
    100%|██████████| 5000/5000 [00:30<00:00, 163.92it/s]




```python
modeldict2=dict(m1=(m1, trace1), m2_nopc = (m2_nopc, trace2_nopc),
              m2_onlyp=(m2_onlyp, trace2_onlyp),
              m2_onlyc=(m2c_onlyc, trace2c_onlyc),
              m2_onlyic=(m2c_onlyic, trace2c_onlyic))
```




```python
names2, models2, traces2=zip(*[(a, b, c) for a, (b, c) in modeldict2.items()])
```




```python
comparedf2 = pm.compare(traces2, models2)
temp=comparedf2.sort_index()
temp['name']=names2
comparedf2 = temp.sort('WAIC').set_index('name')
comparedf2
```


    //anaconda/envs/py35/lib/python3.5/site-packages/ipykernel/__main__.py:4: FutureWarning: sort(columns=....) is deprecated, use sort_values(by=.....)





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>WAIC</th>
      <th>pWAIC</th>
      <th>dWAIC</th>
      <th>weight</th>
      <th>SE</th>
      <th>dSE</th>
      <th>warning</th>
    </tr>
    <tr>
      <th>name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>m2_nopc</th>
      <td>79.1059</td>
      <td>4.22647</td>
      <td>0</td>
      <td>0.61959</td>
      <td>11.0612</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>m1</th>
      <td>80.3046</td>
      <td>5.03686</td>
      <td>1.19871</td>
      <td>0.340258</td>
      <td>11.3985</td>
      <td>0.571957</td>
      <td>1</td>
    </tr>
    <tr>
      <th>m2_onlyp</th>
      <td>84.5787</td>
      <td>3.84888</td>
      <td>5.47276</td>
      <td>0.0401523</td>
      <td>8.98146</td>
      <td>20.1717</td>
      <td>1</td>
    </tr>
    <tr>
      <th>m2_onlyic</th>
      <td>141.327</td>
      <td>8.10745</td>
      <td>62.2212</td>
      <td>1.90956e-14</td>
      <td>31.6664</td>
      <td>338.568</td>
      <td>1</td>
    </tr>
    <tr>
      <th>m2_onlyc</th>
      <td>152.975</td>
      <td>18.1559</td>
      <td>73.8689</td>
      <td>5.64512e-17</td>
      <td>46.6488</td>
      <td>678.014</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



What we find now is that the full-model has much more weight.



```python
compare_plot(comparedf2)
```





    <matplotlib.axes._subplots.AxesSubplot at 0x114153080>




![png](Islands2_files/Islands2_44_1.png)


In either case, our top model excludes the interaction, but the other top model includes it. In the centered case, the non-interacting model has most of the weight, while in the non-centered model, the weights were more equally shared.

In a situation where the interaction model has so much weight, we can say its probably overfit. So in a sense, centering even helps us with our overfitting issues by clearly preferring the non-centered model, as it removes correlation and thus spurious weight being borrowed.

## Computing the posterior predictive

We now write some code to compute the posterior predictive at artbitrary points without having to use theano shared vaiables and sample_ppc, in two different counterfactual situations of low contact and high contact. Since some of our models omit certain terms, we use traces with 0s in them to construct a general function to do this.



```python
def trace_or_zero(trace, name):
    if name in trace.varnames:
        return trace[name]
    else:
        return np.zeros(2*len(trace))
```




```python
trace_or_zero(trace1c, 'alpha')
```





    array([ 3.36384035,  3.36384035,  3.25198963, ...,  3.28698966,
            3.3490696 ,  3.23150147])





```python
from scipy.stats import poisson
def compute_pp(lpgrid, trace, contact=0):
    alphatrace = trace_or_zero(trace, 'alpha')
    betaptrace = trace_or_zero(trace, 'betap')
    betactrace = trace_or_zero(trace, 'betac')
    betapctrace = trace_or_zero(trace, 'betapc')
    tl=2*len(trace)
    gl=lpgrid.shape[0]
    lam = np.empty((gl, tl))
    lpgrid = lpgrid - lpgrid.mean()
    for i, v in enumerate(lpgrid):
        temp = alphatrace + betaptrace*lpgrid[i] + betactrace*contact + betapctrace*contact*lpgrid[i]
        lam[i,:] = poisson.rvs(np.exp(temp))
    return lam
```


We compute the posterior predictive in the counterfactual cases: remember what we are doing here is turning on and off a feature.



```python
lpgrid = np.linspace(6,13,30)
pplow = compute_pp(lpgrid, trace1c)
pphigh = compute_pp(lpgrid, trace1c, contact=1)
```


We compute the medians and the hpds, and plot these against the data



```python
pplowmed = np.median(pplow, axis=1)
pplowhpd = pm.stats.hpd(pplow.T)
pphighmed = np.median(pphigh, axis=1)
pphighhpd = pm.stats.hpd(pphigh.T)
```




```python
plt.plot(df[df['clevel']==1].logpop, df[df['clevel']==1].total_tools,'o', color="g")
plt.plot(df[df['clevel']==0].logpop, df[df['clevel']==0].total_tools,'o', color="r")
plt.plot(lpgrid, pphighmed, color="g")
plt.fill_between(lpgrid, pphighhpd[:,0], pphighhpd[:,1], color="g", alpha=0.2)
plt.plot(lpgrid, pplowmed, color="r")
plt.fill_between(lpgrid, pplowhpd[:,0], pplowhpd[:,1], color="r", alpha=0.2)

```





    <matplotlib.collections.PolyCollection at 0x1262ab400>




![png](Islands2_files/Islands2_54_1.png)


This is for the full centered model. The high contact predictive and data is in green. We undertake this exercise as a prelude to ensembling the models with high Akaike weights

## Ensembling

Ensembles are a good way to combine models where one model may be good at something and the other at something else. Ensembles also help with overfitting if the variance cancels out between the ensemble members: they would all probably overfit in slightly different ways. Lets write a function to do our ensembling for us.



```python
def ensemble(grid, modeldict, comparedf, modelnames, contact=0):
    accum_pp=0
    accum_weight=0
    for m in modelnames:
        weight = comparedf.ix[m]['weight']
        pp = compute_pp(grid, modeldict[m][1], contact)
        print(m, weight, np.median(pp))
        accum_pp += pp*weight
        accum_weight +=weight
    return accum_pp/accum_weight
        
```




```python
ens_pp_low = ensemble(lpgrid, modeldict, comparedf, ['m1c', 'm2c_nopc', 'm2c_onlyp'])
```


    m1c 0.0890866148271 28.0
    m2c_nopc 0.84632720633 28.0
    m2c_onlyp 0.0645861788431 33.0




```python
ens_pp_high = ensemble(lpgrid, modeldict, comparedf, ['m1c', 'm2c_nopc', 'm2c_onlyp'], contact=1)
```


    m1c 0.0890866148271 37.0
    m2c_nopc 0.84632720633 37.0
    m2c_onlyp 0.0645861788431 33.0




```python
pplowmed = np.median(ens_pp_low, axis=1)
pplowhpd = pm.stats.hpd(ens_pp_low.T)
pphighmed = np.median(ens_pp_high, axis=1)
pphighhpd = pm.stats.hpd(ens_pp_high.T)
plt.plot(df[df['clevel']==1].logpop, df[df['clevel']==1].total_tools,'o', color="g")
plt.plot(df[df['clevel']==0].logpop, df[df['clevel']==0].total_tools,'o', color="r")
plt.plot(lpgrid, pphighmed, color="g")
plt.fill_between(lpgrid, pphighhpd[:,0], pphighhpd[:,1], color="g", alpha=0.2)
plt.plot(lpgrid, pplowmed, color="r")
plt.fill_between(lpgrid, pplowhpd[:,0], pplowhpd[:,1], color="r", alpha=0.2)

```





    <matplotlib.collections.PolyCollection at 0x1275685f8>




![png](Islands2_files/Islands2_60_1.png)


The ensemble gives sensible limits and even regularizes down the green band at high population by giving more weight to the no-interaction model.

## Hierarchical Modelling

**Overdispersion** is a problem one finds in most poisson models where the variance of the data is larger than the mean, which is the constraint the poisson distribution imposes.

To simplify things, let us consider here, only the model with log(population). Since there is no contact variable, there are no counterfactual plots and we can view the posterior predictive.



```python
ppsamps = compute_pp(lpgrid, trace2c_onlyp)
ppmed = np.median(ppsamps, axis=1)
pphpd = pm.stats.hpd(ppsamps.T)
plt.plot(df[df['clevel']==1].logpop, df[df['clevel']==1].total_tools,'o', color="g")
plt.plot(df[df['clevel']==0].logpop, df[df['clevel']==0].total_tools,'o', color="r")
plt.plot(lpgrid, ppmed, color="b")
plt.fill_between(lpgrid, pphpd[:,0], pphpd[:,1], color="b", alpha=0.1)
plt.ylim([0, 300])
```





    (0, 300)




![png](Islands2_files/Islands2_63_1.png)


By taking the ratio of the posterior-predictive variance to the posterior-predictive mean, we see that the model is overdispersed.



```python
ppvar=np.var(ppsamps, axis=1)
ppmean=np.mean(ppsamps, axis=1)
```




```python
ppvar/ppmean
```





    array([ 1.28450637,  1.2546132 ,  1.23938696,  1.23189666,  1.2442369 ,
            1.20259161,  1.18695212,  1.19990685,  1.15095771,  1.15486402,
            1.146277  ,  1.14248799,  1.12651325,  1.11379967,  1.11639962,
            1.07987836,  1.10113281,  1.1196618 ,  1.12628776,  1.13029754,
            1.1276347 ,  1.20407582,  1.21529928,  1.23939564,  1.32571464,
            1.39937775,  1.43981034,  1.58385778,  1.70222202,  1.81490436])



Overdispersion can be fixed by considering a mixture model. We shall see this next week. But hierarchical modelling is also a great way to do this.

### Varying Intercepts hierarchical model

What we are basically doing is splitting the intercept into a value constant across the societies and a residual which is society dependent. It is this residual that we will assume is draen from a gaussian with 0 mean and `sigmasoc` ($\sigma_{society}$) standard deviation. Since there is a varying intercept for **every** observation, $\sigma_{society}$ lands up as an estimate of overdispersion amongst societies.



```python
with pm.Model() as m3c:
    betap = pm.Normal("betap", 0, 1)
    alpha = pm.Normal("alpha", 0, 100)
    sigmasoc = pm.HalfCauchy("sigmasoc", 1)
    alphasoc = pm.Normal("alphasoc", 0, sigmasoc, shape=df.shape[0])
    loglam = alpha + alphasoc + betap*df.logpop_c 
    y = pm.Poisson("ntools", mu=t.exp(loglam), observed=df.total_tools)
```


    




```python
with m3c:
    trace3 = pm.sample(5000, njobs=2)
```


    
    
      0%|          | 0/200000 [00:00<?, ?it/s][A[A
    
      1%|          | 1029/200000 [00:00<00:19, 10288.46it/s][A[A
    
      1%|          | 2048/200000 [00:00<00:19, 10257.58it/s][A[A
    
      1%|▏         | 2978/200000 [00:00<00:19, 9949.35it/s] [A[A
    
      2%|▏         | 3723/200000 [00:00<00:21, 9037.11it/s][A[A
    
      2%|▏         | 4426/200000 [00:00<00:23, 8322.71it/s][A[A
    
      3%|▎         | 5249/200000 [00:00<00:23, 8294.23it/s][A[A
    
      3%|▎         | 6031/200000 [00:00<00:23, 8144.42it/s][A[A
    
      3%|▎         | 6897/200000 [00:00<00:23, 8292.30it/s][A[A
    
      4%|▍         | 7803/200000 [00:00<00:22, 8507.20it/s][A[A
    
      4%|▍         | 8747/200000 [00:01<00:21, 8766.68it/s][A[A
    
      5%|▍         | 9663/200000 [00:01<00:21, 8879.61it/s][A[A
    
      5%|▌         | 10598/200000 [00:01<00:21, 9014.22it/s][A[A
    
      6%|▌         | 11529/200000 [00:01<00:20, 9099.70it/s][A[A
    
      6%|▋         | 12634/200000 [00:01<00:19, 9606.79it/s][A[A
    
      7%|▋         | 13598/200000 [00:01<00:19, 9476.46it/s][A[A
    
      7%|▋         | 14585/200000 [00:01<00:19, 9577.08it/s][A[A
    
      8%|▊         | 15545/200000 [00:01<00:19, 9223.59it/s][A[A
    
      8%|▊         | 16472/200000 [00:01<00:20, 9174.05it/s][A[A
    
      9%|▊         | 17431/200000 [00:01<00:19, 9288.81it/s][A[A
    
      9%|▉         | 18363/200000 [00:02<00:21, 8610.86it/s][A[A
    
     10%|▉         | 19236/200000 [00:02<00:24, 7406.91it/s][A[A
    
    Average ELBO = -319.16:  10%|█         | 20015/200000 [00:02<00:27, 6535.69it/s][A[A
    
    Average ELBO = -319.16:  10%|█         | 20714/200000 [00:02<00:30, 5888.36it/s][A[A
    
    Average ELBO = -319.16:  11%|█         | 21435/200000 [00:02<00:28, 6230.11it/s][A[A
    
    Average ELBO = -319.16:  11%|█         | 22451/200000 [00:02<00:25, 7047.47it/s][A[A
    
    Average ELBO = -319.16:  12%|█▏        | 23365/200000 [00:02<00:23, 7567.08it/s][A[A
    
    Average ELBO = -319.16:  12%|█▏        | 24246/200000 [00:02<00:22, 7901.17it/s][A[A
    
    Average ELBO = -319.16:  13%|█▎        | 25078/200000 [00:03<00:23, 7590.58it/s][A[A
    
    Average ELBO = -319.16:  13%|█▎        | 25869/200000 [00:03<00:25, 6781.32it/s][A[A
    
    Average ELBO = -319.16:  13%|█▎        | 26586/200000 [00:03<00:25, 6841.47it/s][A[A
    
    Average ELBO = -319.16:  14%|█▎        | 27298/200000 [00:03<00:25, 6718.83it/s][A[A
    
    Average ELBO = -319.16:  14%|█▍        | 27990/200000 [00:03<00:25, 6754.19it/s][A[A
    
    Average ELBO = -319.16:  14%|█▍        | 28717/200000 [00:03<00:24, 6899.57it/s][A[A
    
    Average ELBO = -319.16:  15%|█▍        | 29418/200000 [00:03<00:27, 6248.57it/s][A[A
    
    Average ELBO = -319.16:  15%|█▌        | 30062/200000 [00:03<00:28, 5989.36it/s][A[A
    
    Average ELBO = -319.16:  15%|█▌        | 30677/200000 [00:03<00:28, 6001.09it/s][A[A
    
    Average ELBO = -319.16:  16%|█▌        | 31566/200000 [00:04<00:25, 6648.44it/s][A[A
    
    Average ELBO = -319.16:  16%|█▋        | 32513/200000 [00:04<00:22, 7299.81it/s][A[A
    
    Average ELBO = -319.16:  17%|█▋        | 33362/200000 [00:04<00:21, 7618.77it/s][A[A
    
    Average ELBO = -319.16:  17%|█▋        | 34282/200000 [00:04<00:20, 8031.43it/s][A[A
    
    Average ELBO = -319.16:  18%|█▊        | 35114/200000 [00:04<00:20, 8033.48it/s][A[A
    
    Average ELBO = -319.16:  18%|█▊        | 35937/200000 [00:04<00:22, 7298.00it/s][A[A
    
    Average ELBO = -319.16:  18%|█▊        | 36694/200000 [00:04<00:23, 6959.71it/s][A[A
    
    Average ELBO = -319.16:  19%|█▉        | 37632/200000 [00:04<00:21, 7543.14it/s][A[A
    
    Average ELBO = -319.16:  19%|█▉        | 38484/200000 [00:04<00:20, 7811.73it/s][A[A
    
    Average ELBO = -319.16:  20%|█▉        | 39442/200000 [00:05<00:19, 8269.14it/s][A[A
    
    Average ELBO = -66.551:  20%|██        | 40397/200000 [00:05<00:18, 8614.32it/s][A[A
    
    Average ELBO = -66.551:  21%|██        | 41342/200000 [00:05<00:17, 8849.03it/s][A[A
    
    Average ELBO = -66.551:  21%|██        | 42340/200000 [00:05<00:17, 9159.00it/s][A[A
    
    Average ELBO = -66.551:  22%|██▏       | 43270/200000 [00:05<00:17, 9001.07it/s][A[A
    
    Average ELBO = -66.551:  22%|██▏       | 44299/200000 [00:05<00:16, 9351.17it/s][A[A
    
    Average ELBO = -66.551:  23%|██▎       | 45245/200000 [00:05<00:16, 9201.95it/s][A[A
    
    Average ELBO = -66.551:  23%|██▎       | 46203/200000 [00:05<00:16, 9309.89it/s][A[A
    
    Average ELBO = -66.551:  24%|██▎       | 47157/200000 [00:05<00:16, 9374.54it/s][A[A
    
    Average ELBO = -66.551:  24%|██▍       | 48099/200000 [00:05<00:16, 9028.57it/s][A[A
    
    Average ELBO = -66.551:  25%|██▍       | 49017/200000 [00:06<00:16, 9071.70it/s][A[A
    
    Average ELBO = -66.551:  25%|██▍       | 49929/200000 [00:06<00:16, 8997.41it/s][A[A
    
    Average ELBO = -66.551:  25%|██▌       | 50896/200000 [00:06<00:16, 9189.11it/s][A[A
    
    Average ELBO = -66.551:  26%|██▌       | 51818/200000 [00:06<00:16, 9138.73it/s][A[A
    
    Average ELBO = -66.551:  26%|██▋       | 52735/200000 [00:06<00:16, 8881.58it/s][A[A
    
    Average ELBO = -66.551:  27%|██▋       | 53627/200000 [00:06<00:16, 8624.52it/s][A[A
    
    Average ELBO = -66.551:  27%|██▋       | 54633/200000 [00:06<00:16, 9009.61it/s][A[A
    
    Average ELBO = -66.551:  28%|██▊       | 55542/200000 [00:06<00:16, 8919.79it/s][A[A
    
    Average ELBO = -66.551:  28%|██▊       | 56440/200000 [00:06<00:18, 7901.03it/s][A[A
    
    Average ELBO = -66.551:  29%|██▊       | 57256/200000 [00:07<00:20, 6876.78it/s][A[A
    
    Average ELBO = -66.551:  29%|██▉       | 57986/200000 [00:07<00:22, 6309.11it/s][A[A
    
    Average ELBO = -66.551:  29%|██▉       | 58656/200000 [00:07<00:22, 6208.12it/s][A[A
    
    Average ELBO = -66.551:  30%|██▉       | 59457/200000 [00:07<00:21, 6657.22it/s][A[A
    
    Average ELBO = -63.734:  30%|███       | 60301/200000 [00:07<00:19, 7106.85it/s][A[A
    
    Average ELBO = -63.734:  31%|███       | 61242/200000 [00:07<00:18, 7669.27it/s][A[A
    
    Average ELBO = -63.734:  31%|███       | 62245/200000 [00:07<00:16, 8242.72it/s][A[A
    
    Average ELBO = -63.734:  32%|███▏      | 63232/200000 [00:07<00:15, 8670.26it/s][A[A
    
    Average ELBO = -63.734:  32%|███▏      | 64173/200000 [00:07<00:15, 8879.30it/s][A[A
    
    Average ELBO = -63.734:  33%|███▎      | 65084/200000 [00:08<00:16, 8251.28it/s][A[A
    
    Average ELBO = -63.734:  33%|███▎      | 65934/200000 [00:08<00:17, 7540.80it/s][A[A
    
    Average ELBO = -63.734:  33%|███▎      | 66718/200000 [00:08<00:19, 6849.53it/s][A[A
    
    Average ELBO = -63.734:  34%|███▎      | 67436/200000 [00:08<00:20, 6571.84it/s][A[A
    
    Average ELBO = -63.734:  34%|███▍      | 68119/200000 [00:08<00:21, 5996.59it/s][A[A
    
    Average ELBO = -63.734:  34%|███▍      | 68771/200000 [00:08<00:21, 6143.37it/s][A[A
    
    Average ELBO = -63.734:  35%|███▍      | 69406/200000 [00:08<00:21, 6138.43it/s][A[A
    
    Average ELBO = -63.734:  35%|███▌      | 70034/200000 [00:08<00:21, 6147.41it/s][A[A
    
    Average ELBO = -63.734:  35%|███▌      | 70725/200000 [00:08<00:20, 6299.52it/s][A[A
    
    Average ELBO = -63.734:  36%|███▌      | 71363/200000 [00:09<00:21, 6109.61it/s][A[A
    
    Average ELBO = -63.734:  36%|███▌      | 71981/200000 [00:09<00:21, 6020.80it/s][A[A
    
    Average ELBO = -63.734:  36%|███▋      | 72790/200000 [00:09<00:19, 6520.50it/s][A[A
    
    Average ELBO = -63.734:  37%|███▋      | 73801/200000 [00:09<00:17, 7294.05it/s][A[A
    
    Average ELBO = -63.734:  37%|███▋      | 74767/200000 [00:09<00:15, 7871.82it/s][A[A
    
    Average ELBO = -63.734:  38%|███▊      | 75817/200000 [00:09<00:14, 8509.53it/s][A[A
    
    Average ELBO = -63.734:  38%|███▊      | 76806/200000 [00:09<00:13, 8880.60it/s][A[A
    
    Average ELBO = -63.734:  39%|███▉      | 77811/200000 [00:09<00:13, 9188.99it/s][A[A
    
    Average ELBO = -63.734:  39%|███▉      | 78758/200000 [00:09<00:14, 8644.85it/s][A[A
    
    Average ELBO = -63.734:  40%|███▉      | 79649/200000 [00:10<00:14, 8248.17it/s][A[A
    
    Average ELBO = -58.669:  40%|████      | 80693/200000 [00:10<00:13, 8801.16it/s][A[A
    
    Average ELBO = -58.669:  41%|████      | 81753/200000 [00:10<00:12, 9272.03it/s][A[A
    
    Average ELBO = -58.669:  41%|████▏     | 82736/200000 [00:10<00:12, 9432.25it/s][A[A
    
    Average ELBO = -58.669:  42%|████▏     | 83772/200000 [00:10<00:11, 9691.61it/s][A[A
    
    Average ELBO = -58.669:  42%|████▏     | 84756/200000 [00:10<00:12, 9328.44it/s][A[A
    
    Average ELBO = -58.669:  43%|████▎     | 85804/200000 [00:10<00:11, 9646.03it/s][A[A
    
    Average ELBO = -58.669:  43%|████▎     | 86865/200000 [00:10<00:11, 9910.18it/s][A[A
    
    Average ELBO = -58.669:  44%|████▍     | 87867/200000 [00:10<00:11, 9546.59it/s][A[A
    
    Average ELBO = -58.669:  44%|████▍     | 88832/200000 [00:11<00:12, 8602.54it/s][A[A
    
    Average ELBO = -58.669:  45%|████▍     | 89717/200000 [00:11<00:14, 7714.93it/s][A[A
    
    Average ELBO = -58.669:  45%|████▌     | 90523/200000 [00:11<00:14, 7749.69it/s][A[A
    
    Average ELBO = -58.669:  46%|████▌     | 91542/200000 [00:11<00:12, 8348.26it/s][A[A
    
    Average ELBO = -58.669:  46%|████▋     | 92639/200000 [00:11<00:11, 8991.76it/s][A[A
    
    Average ELBO = -58.669:  47%|████▋     | 93662/200000 [00:11<00:11, 9330.24it/s][A[A
    
    Average ELBO = -58.669:  47%|████▋     | 94623/200000 [00:11<00:11, 9144.94it/s][A[A
    
    Average ELBO = -58.669:  48%|████▊     | 95558/200000 [00:11<00:12, 8194.67it/s][A[A
    
    Average ELBO = -58.669:  48%|████▊     | 96411/200000 [00:11<00:13, 7859.37it/s][A[A
    
    Average ELBO = -58.669:  49%|████▊     | 97223/200000 [00:12<00:13, 7497.23it/s][A[A
    
    Average ELBO = -58.669:  49%|████▉     | 97995/200000 [00:12<00:13, 7359.73it/s][A[A
    
    Average ELBO = -58.669:  49%|████▉     | 98851/200000 [00:12<00:13, 7682.44it/s][A[A
    
    Average ELBO = -58.669:  50%|████▉     | 99892/200000 [00:12<00:12, 8336.10it/s][A[A
    
    Average ELBO = -51.089:  50%|█████     | 100862/200000 [00:12<00:11, 8701.56it/s][A[A
    
    Average ELBO = -51.089:  51%|█████     | 101793/200000 [00:12<00:11, 8874.61it/s][A[A
    
    Average ELBO = -51.089:  51%|█████▏    | 102698/200000 [00:12<00:11, 8116.13it/s][A[A
    
    Average ELBO = -51.089:  52%|█████▏    | 103534/200000 [00:12<00:12, 7716.96it/s][A[A
    
    Average ELBO = -51.089:  52%|█████▏    | 104327/200000 [00:12<00:12, 7637.82it/s][A[A
    
    Average ELBO = -51.089:  53%|█████▎    | 105106/200000 [00:13<00:12, 7671.49it/s][A[A
    
    Average ELBO = -51.089:  53%|█████▎    | 106124/200000 [00:13<00:11, 8282.90it/s][A[A
    
    Average ELBO = -51.089:  53%|█████▎    | 106974/200000 [00:13<00:11, 7761.17it/s][A[A
    
    Average ELBO = -51.089:  54%|█████▍    | 107772/200000 [00:13<00:12, 7387.90it/s][A[A
    
    Average ELBO = -51.089:  54%|█████▍    | 108530/200000 [00:13<00:12, 7217.58it/s][A[A
    
    Average ELBO = -51.089:  55%|█████▍    | 109335/200000 [00:13<00:12, 7448.28it/s][A[A
    
    Average ELBO = -51.089:  55%|█████▌    | 110399/200000 [00:13<00:10, 8184.74it/s][A[A
    
    Average ELBO = -51.089:  56%|█████▌    | 111255/200000 [00:13<00:10, 8292.29it/s][A[A
    
    Average ELBO = -51.089:  56%|█████▌    | 112255/200000 [00:13<00:10, 8739.69it/s][A[A
    
    Average ELBO = -51.089:  57%|█████▋    | 113151/200000 [00:13<00:10, 8674.80it/s][A[A
    
    Average ELBO = -51.089:  57%|█████▋    | 114069/200000 [00:14<00:09, 8815.95it/s][A[A
    
    Average ELBO = -51.089:  58%|█████▊    | 115022/200000 [00:14<00:09, 9013.92it/s][A[A
    
    Average ELBO = -51.089:  58%|█████▊    | 116093/200000 [00:14<00:08, 9459.94it/s][A[A
    
    Average ELBO = -51.089:  59%|█████▊    | 117099/200000 [00:14<00:08, 9623.16it/s][A[A
    
    Average ELBO = -51.089:  59%|█████▉    | 118149/200000 [00:14<00:08, 9864.59it/s][A[A
    
    Average ELBO = -51.089:  60%|█████▉    | 119144/200000 [00:14<00:10, 7956.99it/s][A[A
    
    Average ELBO = -49.861:  60%|██████    | 120058/200000 [00:14<00:09, 8278.17it/s][A[A
    
    Average ELBO = -49.861:  60%|██████    | 120958/200000 [00:14<00:09, 8482.11it/s][A[A
    
    Average ELBO = -49.861:  61%|██████    | 121962/200000 [00:14<00:08, 8894.34it/s][A[A
    
    Average ELBO = -49.861:  61%|██████▏   | 122882/200000 [00:15<00:09, 8096.92it/s][A[A
    
    Average ELBO = -49.861:  62%|██████▏   | 123727/200000 [00:15<00:10, 7575.49it/s][A[A
    
    Average ELBO = -49.861:  62%|██████▏   | 124516/200000 [00:15<00:10, 7033.06it/s][A[A
    
    Average ELBO = -49.861:  63%|██████▎   | 125249/200000 [00:15<00:11, 6772.20it/s][A[A
    
    Average ELBO = -49.861:  63%|██████▎   | 125957/200000 [00:15<00:10, 6860.27it/s][A[A
    
    Average ELBO = -49.861:  63%|██████▎   | 126660/200000 [00:15<00:10, 6855.64it/s][A[A
    
    Average ELBO = -49.861:  64%|██████▎   | 127357/200000 [00:15<00:10, 6851.54it/s][A[A
    
    Average ELBO = -49.861:  64%|██████▍   | 128051/200000 [00:15<00:10, 6620.34it/s][A[A
    
    Average ELBO = -49.861:  64%|██████▍   | 128721/200000 [00:15<00:10, 6502.27it/s][A[A
    
    Average ELBO = -49.861:  65%|██████▍   | 129465/200000 [00:16<00:10, 6757.17it/s][A[A
    
    Average ELBO = -49.861:  65%|██████▌   | 130202/200000 [00:16<00:10, 6897.08it/s][A[A
    
    Average ELBO = -49.861:  65%|██████▌   | 130897/200000 [00:16<00:10, 6755.66it/s][A[A
    
    Average ELBO = -49.861:  66%|██████▌   | 131577/200000 [00:16<00:10, 6686.75it/s][A[A
    
    Average ELBO = -49.861:  66%|██████▌   | 132249/200000 [00:16<00:10, 6445.66it/s][A[A
    
    Average ELBO = -49.861:  66%|██████▋   | 132898/200000 [00:16<00:11, 5977.15it/s][A[A
    
    Average ELBO = -49.861:  67%|██████▋   | 133628/200000 [00:16<00:10, 6319.89it/s][A[A
    
    Average ELBO = -49.861:  67%|██████▋   | 134384/200000 [00:16<00:09, 6645.76it/s][A[A
    
    Average ELBO = -49.861:  68%|██████▊   | 135127/200000 [00:16<00:09, 6860.94it/s][A[A
    
    Average ELBO = -49.861:  68%|██████▊   | 135878/200000 [00:17<00:09, 7042.29it/s][A[A
    
    Average ELBO = -49.861:  68%|██████▊   | 136654/200000 [00:17<00:08, 7242.50it/s][A[A
    
    Average ELBO = -49.861:  69%|██████▊   | 137386/200000 [00:17<00:08, 7129.58it/s][A[A
    
    Average ELBO = -49.861:  69%|██████▉   | 138105/200000 [00:17<00:08, 6940.77it/s][A[A
    
    Average ELBO = -49.861:  69%|██████▉   | 138805/200000 [00:17<00:09, 6432.65it/s][A[A
    
    Average ELBO = -49.861:  70%|██████▉   | 139460/200000 [00:17<00:09, 6420.48it/s][A[A
    
    Average ELBO = -49.835:  70%|███████   | 140206/200000 [00:17<00:08, 6699.05it/s][A[A
    
    Average ELBO = -49.835:  70%|███████   | 140947/200000 [00:17<00:08, 6897.14it/s][A[A
    
    Average ELBO = -49.835:  71%|███████   | 141645/200000 [00:17<00:08, 6613.88it/s][A[A
    
    Average ELBO = -49.835:  71%|███████   | 142323/200000 [00:18<00:08, 6662.82it/s][A[A
    
    Average ELBO = -49.835:  71%|███████▏  | 142995/200000 [00:18<00:08, 6617.34it/s][A[A
    
    Average ELBO = -49.835:  72%|███████▏  | 143692/200000 [00:18<00:08, 6719.08it/s][A[A
    
    Average ELBO = -49.835:  72%|███████▏  | 144368/200000 [00:18<00:08, 6485.62it/s][A[A
    
    Average ELBO = -49.835:  73%|███████▎  | 145058/200000 [00:18<00:08, 6601.50it/s][A[A
    
    Average ELBO = -49.835:  73%|███████▎  | 145722/200000 [00:18<00:08, 6212.81it/s][A[A
    
    Average ELBO = -49.835:  73%|███████▎  | 146367/200000 [00:18<00:08, 6281.03it/s][A[A
    
    Average ELBO = -49.835:  74%|███████▎  | 147019/200000 [00:18<00:08, 6350.48it/s][A[A
    
    Average ELBO = -49.835:  74%|███████▍  | 147658/200000 [00:18<00:08, 6248.58it/s][A[A
    
    Average ELBO = -49.835:  74%|███████▍  | 148368/200000 [00:18<00:07, 6469.52it/s][A[A
    
    Average ELBO = -49.835:  75%|███████▍  | 149038/200000 [00:19<00:07, 6536.38it/s][A[A
    
    Average ELBO = -49.835:  75%|███████▍  | 149718/200000 [00:19<00:07, 6611.51it/s][A[A
    
    Average ELBO = -49.835:  75%|███████▌  | 150408/200000 [00:19<00:07, 6693.64it/s][A[A
    
    Average ELBO = -49.835:  76%|███████▌  | 151080/200000 [00:19<00:07, 6627.86it/s][A[A
    
    Average ELBO = -49.835:  76%|███████▌  | 151745/200000 [00:19<00:07, 6545.80it/s][A[A
    
    Average ELBO = -49.835:  76%|███████▌  | 152401/200000 [00:19<00:07, 6128.64it/s][A[A
    
    Average ELBO = -49.835:  77%|███████▋  | 153025/200000 [00:19<00:07, 6159.55it/s][A[A
    
    Average ELBO = -49.835:  77%|███████▋  | 153653/200000 [00:19<00:07, 6191.00it/s][A[A
    
    Average ELBO = -49.835:  77%|███████▋  | 154276/200000 [00:19<00:07, 5973.85it/s][A[A
    
    Average ELBO = -49.835:  77%|███████▋  | 154878/200000 [00:20<00:07, 5908.91it/s][A[A
    
    Average ELBO = -49.835:  78%|███████▊  | 155504/200000 [00:20<00:07, 6001.62it/s][A[A
    
    Average ELBO = -49.835:  78%|███████▊  | 156141/200000 [00:20<00:07, 6076.69it/s][A[A
    
    Average ELBO = -49.835:  78%|███████▊  | 156759/200000 [00:20<00:07, 6106.09it/s][A[A
    
    Average ELBO = -49.835:  79%|███████▊  | 157401/200000 [00:20<00:06, 6193.87it/s][A[A
    
    Average ELBO = -49.835:  79%|███████▉  | 158022/200000 [00:20<00:07, 5556.26it/s][A[A
    
    Average ELBO = -49.835:  79%|███████▉  | 158591/200000 [00:20<00:07, 5533.98it/s][A[A
    
    Average ELBO = -49.835:  80%|███████▉  | 159156/200000 [00:20<00:07, 5567.85it/s][A[A
    
    Average ELBO = -49.835:  80%|███████▉  | 159744/200000 [00:20<00:07, 5657.50it/s][A[A
    
    Average ELBO = -49.865:  80%|████████  | 160544/200000 [00:20<00:06, 6201.30it/s][A[A
    
    Average ELBO = -49.865:  81%|████████  | 161365/200000 [00:21<00:05, 6691.47it/s][A[A
    
    Average ELBO = -49.865:  81%|████████  | 162356/200000 [00:21<00:05, 7412.67it/s][A[A
    
    Average ELBO = -49.865:  82%|████████▏ | 163139/200000 [00:21<00:04, 7419.22it/s][A[A
    
    Average ELBO = -49.865:  82%|████████▏ | 163910/200000 [00:21<00:05, 6746.91it/s][A[A
    
    Average ELBO = -49.865:  82%|████████▏ | 164617/200000 [00:21<00:05, 5957.87it/s][A[A
    
    Average ELBO = -49.865:  83%|████████▎ | 165253/200000 [00:21<00:06, 5645.65it/s][A[A
    
    Average ELBO = -49.865:  83%|████████▎ | 165849/200000 [00:21<00:05, 5695.45it/s][A[A
    
    Average ELBO = -49.865:  83%|████████▎ | 166533/200000 [00:21<00:05, 5993.41it/s][A[A
    
    Average ELBO = -49.865:  84%|████████▎ | 167179/200000 [00:21<00:05, 6125.43it/s][A[A
    
    Average ELBO = -49.865:  84%|████████▍ | 167806/200000 [00:22<00:05, 5883.14it/s][A[A
    
    Average ELBO = -49.865:  84%|████████▍ | 168500/200000 [00:22<00:05, 6162.96it/s][A[A
    
    Average ELBO = -49.865:  85%|████████▍ | 169207/200000 [00:22<00:04, 6409.17it/s][A[A
    
    Average ELBO = -49.865:  85%|████████▍ | 169859/200000 [00:22<00:04, 6220.23it/s][A[A
    
    Average ELBO = -49.865:  85%|████████▌ | 170490/200000 [00:22<00:04, 6010.35it/s][A[A
    
    Average ELBO = -49.865:  86%|████████▌ | 171157/200000 [00:22<00:04, 6192.97it/s][A[A
    
    Average ELBO = -49.865:  86%|████████▌ | 171784/200000 [00:22<00:04, 6095.93it/s][A[A
    
    Average ELBO = -49.865:  86%|████████▌ | 172399/200000 [00:22<00:04, 5913.94it/s][A[A
    
    Average ELBO = -49.865:  86%|████████▋ | 172996/200000 [00:22<00:04, 5801.20it/s][A[A
    
    Average ELBO = -49.865:  87%|████████▋ | 173580/200000 [00:23<00:04, 5677.93it/s][A[A
    
    Average ELBO = -49.865:  87%|████████▋ | 174180/200000 [00:23<00:04, 5770.51it/s][A[A
    
    Average ELBO = -49.865:  87%|████████▋ | 174761/200000 [00:23<00:04, 5760.91it/s][A[A
    
    Average ELBO = -49.865:  88%|████████▊ | 175390/200000 [00:23<00:04, 5905.81it/s][A[A
    
    Average ELBO = -49.865:  88%|████████▊ | 175983/200000 [00:23<00:04, 5828.36it/s][A[A
    
    Average ELBO = -49.865:  88%|████████▊ | 176568/200000 [00:23<00:04, 5598.56it/s][A[A
    
    Average ELBO = -49.865:  89%|████████▊ | 177132/200000 [00:23<00:04, 5508.06it/s][A[A
    
    Average ELBO = -49.865:  89%|████████▉ | 177702/200000 [00:23<00:04, 5560.31it/s][A[A
    
    Average ELBO = -49.865:  89%|████████▉ | 178260/200000 [00:23<00:03, 5524.15it/s][A[A
    
    Average ELBO = -49.865:  89%|████████▉ | 178829/200000 [00:23<00:03, 5566.50it/s][A[A
    
    Average ELBO = -49.865:  90%|████████▉ | 179479/200000 [00:24<00:03, 5816.27it/s][A[A
    
    Average ELBO = -49.848:  90%|█████████ | 180116/200000 [00:24<00:03, 5971.91it/s][A[A
    
    Average ELBO = -49.848:  90%|█████████ | 180717/200000 [00:24<00:03, 5799.09it/s][A[A
    
    Average ELBO = -49.848:  91%|█████████ | 181347/200000 [00:24<00:03, 5940.67it/s][A[A
    
    Average ELBO = -49.848:  91%|█████████ | 181959/200000 [00:24<00:03, 5990.96it/s][A[A
    
    Average ELBO = -49.848:  91%|█████████▏| 182561/200000 [00:24<00:03, 5373.62it/s][A[A
    
    Average ELBO = -49.848:  92%|█████████▏| 183132/200000 [00:24<00:03, 5454.18it/s][A[A
    
    Average ELBO = -49.848:  92%|█████████▏| 183688/200000 [00:24<00:02, 5452.44it/s][A[A
    
    Average ELBO = -49.848:  92%|█████████▏| 184319/200000 [00:24<00:02, 5674.85it/s][A[A
    
    Average ELBO = -49.848:  92%|█████████▏| 184993/200000 [00:25<00:02, 5956.39it/s][A[A
    
    Average ELBO = -49.848:  93%|█████████▎| 185598/200000 [00:25<00:02, 5736.23it/s][A[A
    
    Average ELBO = -49.848:  93%|█████████▎| 186296/200000 [00:25<00:02, 6057.29it/s][A[A
    
    Average ELBO = -49.848:  93%|█████████▎| 186949/200000 [00:25<00:02, 6191.07it/s][A[A
    
    Average ELBO = -49.848:  94%|█████████▍| 187593/200000 [00:25<00:01, 6260.74it/s][A[A
    
    Average ELBO = -49.848:  94%|█████████▍| 188225/200000 [00:25<00:02, 5189.37it/s][A[A
    
    Average ELBO = -49.848:  94%|█████████▍| 188779/200000 [00:25<00:02, 4754.94it/s][A[A
    
    Average ELBO = -49.848:  95%|█████████▍| 189286/200000 [00:25<00:02, 4689.06it/s][A[A
    
    Average ELBO = -49.848:  95%|█████████▍| 189778/200000 [00:25<00:02, 4704.99it/s][A[A
    
    Average ELBO = -49.848:  95%|█████████▌| 190465/200000 [00:26<00:01, 5182.57it/s][A[A
    
    Average ELBO = -49.848:  96%|█████████▌| 191302/200000 [00:26<00:01, 5847.08it/s][A[A
    
    Average ELBO = -49.848:  96%|█████████▌| 192196/200000 [00:26<00:01, 6520.25it/s][A[A
    
    Average ELBO = -49.848:  96%|█████████▋| 192966/200000 [00:26<00:01, 6785.80it/s][A[A
    
    Average ELBO = -49.848:  97%|█████████▋| 193688/200000 [00:26<00:00, 6687.31it/s][A[A
    
    Average ELBO = -49.848:  97%|█████████▋| 194388/200000 [00:26<00:00, 6444.68it/s][A[A
    
    Average ELBO = -49.848:  98%|█████████▊| 195056/200000 [00:26<00:00, 6437.03it/s][A[A
    
    Average ELBO = -49.848:  98%|█████████▊| 195732/200000 [00:26<00:00, 6526.95it/s][A[A
    
    Average ELBO = -49.848:  98%|█████████▊| 196397/200000 [00:26<00:00, 6445.53it/s][A[A
    
    Average ELBO = -49.848:  99%|█████████▊| 197050/200000 [00:27<00:00, 6376.40it/s][A[A
    
    Average ELBO = -49.848:  99%|█████████▉| 197694/200000 [00:27<00:00, 6249.94it/s][A[A
    
    Average ELBO = -49.848:  99%|█████████▉| 198324/200000 [00:27<00:00, 5910.70it/s][A[A
    
    Average ELBO = -49.848:  99%|█████████▉| 198975/200000 [00:27<00:00, 6078.18it/s][A[A
    
    Average ELBO = -49.848: 100%|█████████▉| 199916/200000 [00:27<00:00, 6800.35it/s][A[A
    
    Average ELBO = -49.848: 100%|██████████| 200000/200000 [00:27<00:00, 7283.00it/s][A[A
    
      0%|          | 0/5000 [00:00<?, ?it/s][A[A
    
      0%|          | 1/5000 [00:00<09:54,  8.42it/s][A[A
    
      0%|          | 11/5000 [00:00<07:10, 11.59it/s][A[A
    
      1%|          | 26/5000 [00:00<05:10, 16.02it/s][A[A
    
      1%|          | 47/5000 [00:00<03:43, 22.15it/s][A[A
    
      1%|          | 62/5000 [00:00<02:45, 29.75it/s][A[A
    
      2%|▏         | 77/5000 [00:00<02:05, 39.09it/s][A[A
    
      2%|▏         | 97/5000 [00:00<01:35, 51.48it/s][A[A
    
      2%|▏         | 117/5000 [00:00<01:13, 66.17it/s][A[A
    
      3%|▎         | 135/5000 [00:00<00:59, 81.27it/s][A[A
    
      3%|▎         | 152/5000 [00:01<00:51, 95.00it/s][A[A
    
      3%|▎         | 171/5000 [00:01<00:43, 111.10it/s][A[A
    
      4%|▍         | 188/5000 [00:01<00:39, 122.77it/s][A[A
    
      4%|▍         | 212/5000 [00:01<00:33, 143.49it/s][A[A
    
      5%|▍         | 231/5000 [00:01<00:31, 150.27it/s][A[A
    
      5%|▌         | 250/5000 [00:01<00:30, 157.04it/s][A[A
    
      5%|▌         | 271/5000 [00:01<00:27, 169.68it/s][A[A
    
      6%|▌         | 294/5000 [00:01<00:25, 183.68it/s][A[A
    
      6%|▋         | 315/5000 [00:01<00:25, 186.03it/s][A[A
    
      7%|▋         | 336/5000 [00:01<00:24, 191.09it/s][A[A
    
      7%|▋         | 356/5000 [00:02<00:24, 192.57it/s][A[A
    
      8%|▊         | 377/5000 [00:02<00:23, 194.07it/s][A[A
    
      8%|▊         | 397/5000 [00:02<00:24, 190.16it/s][A[A
    
      8%|▊         | 417/5000 [00:02<00:24, 188.25it/s][A[A
    
      9%|▊         | 437/5000 [00:02<00:24, 189.84it/s][A[A
    
      9%|▉         | 457/5000 [00:02<00:24, 183.67it/s][A[A
    
     10%|▉         | 477/5000 [00:02<00:24, 186.31it/s][A[A
    
     10%|█         | 500/5000 [00:02<00:22, 195.91it/s][A[A
    
     10%|█         | 520/5000 [00:02<00:22, 196.03it/s][A[A
    
     11%|█         | 540/5000 [00:03<00:23, 189.00it/s][A[A
    
     11%|█         | 560/5000 [00:03<00:24, 179.67it/s][A[A
    
     12%|█▏        | 580/5000 [00:03<00:23, 184.96it/s][A[A
    
     12%|█▏        | 599/5000 [00:03<00:23, 184.05it/s][A[A
    
     12%|█▏        | 620/5000 [00:03<00:23, 188.74it/s][A[A
    
     13%|█▎        | 639/5000 [00:03<00:23, 188.98it/s][A[A
    
     13%|█▎        | 658/5000 [00:03<00:22, 189.22it/s][A[A
    
     14%|█▎        | 679/5000 [00:03<00:22, 193.61it/s][A[A
    
     14%|█▍        | 701/5000 [00:03<00:21, 198.66it/s][A[A
    
     14%|█▍        | 724/5000 [00:03<00:20, 206.39it/s][A[A
    
     15%|█▍        | 745/5000 [00:04<00:21, 199.01it/s][A[A
    
     15%|█▌        | 766/5000 [00:04<00:21, 198.26it/s][A[A
    
     16%|█▌        | 787/5000 [00:04<00:20, 201.12it/s][A[A
    
     16%|█▌        | 809/5000 [00:04<00:20, 205.48it/s][A[A
    
     17%|█▋        | 830/5000 [00:04<00:20, 205.18it/s][A[A
    
     17%|█▋        | 851/5000 [00:04<00:22, 183.31it/s][A[A
    
     17%|█▋        | 870/5000 [00:04<00:23, 176.45it/s][A[A
    
     18%|█▊        | 889/5000 [00:04<00:22, 178.86it/s][A[A
    
     18%|█▊        | 908/5000 [00:04<00:22, 181.03it/s][A[A
    
     19%|█▊        | 927/5000 [00:05<00:22, 178.43it/s][A[A
    
     19%|█▉        | 945/5000 [00:05<00:23, 173.03it/s][A[A
    
     19%|█▉        | 963/5000 [00:05<00:23, 172.50it/s][A[A
    
     20%|█▉        | 981/5000 [00:05<00:23, 171.21it/s][A[A
    
     20%|█▉        | 999/5000 [00:05<00:25, 158.80it/s][A[A
    
     20%|██        | 1016/5000 [00:05<00:26, 151.72it/s][A[A
    
     21%|██        | 1035/5000 [00:05<00:24, 160.34it/s][A[A
    
     21%|██        | 1053/5000 [00:05<00:23, 164.72it/s][A[A
    
     21%|██▏       | 1070/5000 [00:05<00:24, 161.09it/s][A[A
    
     22%|██▏       | 1087/5000 [00:06<00:25, 150.82it/s][A[A
    
     22%|██▏       | 1103/5000 [00:06<00:25, 151.78it/s][A[A
    
     22%|██▏       | 1119/5000 [00:06<00:25, 153.00it/s][A[A
    
     23%|██▎       | 1137/5000 [00:06<00:24, 155.73it/s][A[A
    
     23%|██▎       | 1159/5000 [00:06<00:22, 170.61it/s][A[A
    
     24%|██▎       | 1177/5000 [00:06<00:22, 169.47it/s][A[A
    
     24%|██▍       | 1195/5000 [00:06<00:22, 167.05it/s][A[A
    
     24%|██▍       | 1212/5000 [00:06<00:22, 164.79it/s][A[A
    
     25%|██▍       | 1229/5000 [00:06<00:22, 165.85it/s][A[A
    
     25%|██▍       | 1246/5000 [00:07<00:23, 161.73it/s][A[A
    
     25%|██▌       | 1263/5000 [00:07<00:24, 153.74it/s][A[A
    
     26%|██▌       | 1282/5000 [00:07<00:22, 162.39it/s][A[A
    
     26%|██▌       | 1299/5000 [00:07<00:23, 158.79it/s][A[A
    
     26%|██▋       | 1316/5000 [00:07<00:23, 153.59it/s][A[A
    
     27%|██▋       | 1332/5000 [00:07<00:25, 144.09it/s][A[A
    
     27%|██▋       | 1347/5000 [00:07<00:25, 142.06it/s][A[A
    
     27%|██▋       | 1362/5000 [00:07<00:26, 136.12it/s][A[A
    
     28%|██▊       | 1376/5000 [00:08<00:26, 134.24it/s][A[A
    
     28%|██▊       | 1390/5000 [00:08<00:28, 126.97it/s][A[A
    
     28%|██▊       | 1403/5000 [00:08<00:29, 122.84it/s][A[A
    
     28%|██▊       | 1418/5000 [00:08<00:27, 129.38it/s][A[A
    
     29%|██▊       | 1433/5000 [00:08<00:26, 133.96it/s][A[A
    
     29%|██▉       | 1447/5000 [00:08<00:28, 125.83it/s][A[A
    
     29%|██▉       | 1460/5000 [00:08<00:28, 124.37it/s][A[A
    
     29%|██▉       | 1473/5000 [00:08<00:29, 119.24it/s][A[A
    
     30%|██▉       | 1487/5000 [00:08<00:28, 123.10it/s][A[A
    
     30%|███       | 1501/5000 [00:09<00:28, 122.30it/s][A[A
    
     30%|███       | 1519/5000 [00:09<00:25, 133.90it/s][A[A
    
     31%|███       | 1533/5000 [00:09<00:26, 131.17it/s][A[A
    
     31%|███       | 1549/5000 [00:09<00:24, 138.18it/s][A[A
    
     31%|███▏      | 1564/5000 [00:09<00:25, 137.21it/s][A[A
    
     32%|███▏      | 1583/5000 [00:09<00:22, 149.31it/s][A[A
    
     32%|███▏      | 1606/5000 [00:09<00:20, 166.26it/s][A[A
    
     32%|███▎      | 1625/5000 [00:09<00:19, 172.05it/s][A[A
    
     33%|███▎      | 1643/5000 [00:09<00:19, 174.02it/s][A[A
    
     33%|███▎      | 1661/5000 [00:09<00:19, 169.69it/s][A[A
    
     34%|███▎      | 1679/5000 [00:10<00:20, 165.02it/s][A[A
    
     34%|███▍      | 1699/5000 [00:10<00:19, 171.81it/s][A[A
    
     34%|███▍      | 1719/5000 [00:10<00:18, 177.63it/s][A[A
    
     35%|███▍      | 1742/5000 [00:10<00:17, 189.53it/s][A[A
    
     35%|███▌      | 1762/5000 [00:10<00:17, 188.03it/s][A[A
    
     36%|███▌      | 1782/5000 [00:10<00:17, 188.54it/s][A[A
    
     36%|███▌      | 1802/5000 [00:10<00:17, 186.03it/s][A[A
    
     36%|███▋      | 1825/5000 [00:10<00:16, 197.34it/s][A[A
    
     37%|███▋      | 1846/5000 [00:10<00:16, 186.40it/s][A[A
    
     37%|███▋      | 1865/5000 [00:11<00:16, 185.16it/s][A[A
    
     38%|███▊      | 1884/5000 [00:11<00:17, 180.87it/s][A[A
    
     38%|███▊      | 1903/5000 [00:11<00:17, 181.03it/s][A[A
    
     38%|███▊      | 1922/5000 [00:11<00:17, 176.41it/s][A[A
    
     39%|███▉      | 1940/5000 [00:11<00:17, 175.30it/s][A[A
    
     39%|███▉      | 1960/5000 [00:11<00:17, 178.02it/s][A[A
    
     40%|███▉      | 1978/5000 [00:11<00:17, 173.65it/s][A[A
    
     40%|███▉      | 1996/5000 [00:11<00:17, 170.87it/s][A[A
    
     40%|████      | 2014/5000 [00:11<00:17, 168.22it/s][A[A
    
     41%|████      | 2031/5000 [00:12<00:17, 167.39it/s][A[A
    
     41%|████      | 2049/5000 [00:12<00:17, 170.50it/s][A[A
    
     41%|████▏     | 2067/5000 [00:12<00:17, 169.96it/s][A[A
    
     42%|████▏     | 2085/5000 [00:12<00:18, 158.23it/s][A[A
    
     42%|████▏     | 2103/5000 [00:12<00:17, 161.97it/s][A[A
    
     42%|████▏     | 2124/5000 [00:12<00:16, 173.16it/s][A[A
    
     43%|████▎     | 2142/5000 [00:12<00:16, 174.81it/s][A[A
    
     43%|████▎     | 2161/5000 [00:12<00:15, 178.43it/s][A[A
    
     44%|████▎     | 2180/5000 [00:12<00:16, 166.38it/s][A[A
    
     44%|████▍     | 2200/5000 [00:13<00:16, 172.59it/s][A[A
    
     44%|████▍     | 2220/5000 [00:13<00:15, 178.36it/s][A[A
    
     45%|████▍     | 2239/5000 [00:13<00:16, 167.95it/s][A[A
    
     45%|████▌     | 2257/5000 [00:13<00:16, 166.95it/s][A[A
    
     46%|████▌     | 2277/5000 [00:13<00:15, 175.56it/s][A[A
    
     46%|████▌     | 2295/5000 [00:13<00:17, 152.09it/s][A[A
    
     46%|████▌     | 2311/5000 [00:13<00:19, 139.98it/s][A[A
    
     47%|████▋     | 2326/5000 [00:13<00:25, 106.07it/s][A[A
    
     47%|████▋     | 2342/5000 [00:14<00:22, 117.37it/s][A[A
    
     47%|████▋     | 2360/5000 [00:14<00:20, 127.38it/s][A[A
    
     48%|████▊     | 2375/5000 [00:14<00:20, 126.13it/s][A[A
    
     48%|████▊     | 2391/5000 [00:14<00:19, 131.51it/s][A[A
    
     48%|████▊     | 2406/5000 [00:14<00:19, 134.70it/s][A[A
    
     48%|████▊     | 2420/5000 [00:14<00:19, 130.29it/s][A[A
    
     49%|████▊     | 2437/5000 [00:14<00:18, 139.18it/s][A[A
    
     49%|████▉     | 2456/5000 [00:14<00:16, 150.46it/s][A[A
    
     49%|████▉     | 2472/5000 [00:14<00:17, 145.71it/s][A[A
    
     50%|████▉     | 2488/5000 [00:15<00:19, 128.51it/s][A[A
    
     50%|█████     | 2502/5000 [00:15<00:22, 111.59it/s][A[A
    
     50%|█████     | 2517/5000 [00:15<00:20, 119.04it/s][A[A
    
     51%|█████     | 2533/5000 [00:15<00:19, 128.05it/s][A[A
    
     51%|█████     | 2547/5000 [00:15<00:18, 129.28it/s][A[A
    
     51%|█████▏    | 2564/5000 [00:15<00:17, 137.79it/s][A[A
    
     52%|█████▏    | 2579/5000 [00:15<00:18, 129.93it/s][A[A
    
     52%|█████▏    | 2596/5000 [00:15<00:17, 136.09it/s][A[A
    
     52%|█████▏    | 2615/5000 [00:16<00:16, 146.95it/s][A[A
    
     53%|█████▎    | 2631/5000 [00:16<00:16, 144.87it/s][A[A
    
     53%|█████▎    | 2646/5000 [00:16<00:16, 144.44it/s][A[A
    
     53%|█████▎    | 2661/5000 [00:16<00:16, 140.78it/s][A[A
    
     54%|█████▎    | 2676/5000 [00:16<00:17, 132.22it/s][A[A
    
     54%|█████▍    | 2691/5000 [00:16<00:16, 137.06it/s][A[A
    
     54%|█████▍    | 2705/5000 [00:16<00:16, 136.45it/s][A[A
    
     54%|█████▍    | 2719/5000 [00:16<00:17, 128.23it/s][A[A
    
     55%|█████▍    | 2735/5000 [00:16<00:16, 134.84it/s][A[A
    
     55%|█████▌    | 2757/5000 [00:17<00:14, 150.50it/s][A[A
    
     55%|█████▌    | 2773/5000 [00:17<00:15, 147.58it/s][A[A
    
     56%|█████▌    | 2789/5000 [00:17<00:15, 146.31it/s][A[A
    
     56%|█████▌    | 2805/5000 [00:17<00:15, 144.77it/s][A[A
    
     56%|█████▋    | 2820/5000 [00:17<00:15, 144.13it/s][A[A
    
     57%|█████▋    | 2838/5000 [00:17<00:14, 152.98it/s][A[A
    
     57%|█████▋    | 2855/5000 [00:17<00:13, 155.90it/s][A[A
    
     57%|█████▋    | 2871/5000 [00:17<00:14, 149.40it/s][A[A
    
     58%|█████▊    | 2887/5000 [00:17<00:15, 135.62it/s][A[A
    
     58%|█████▊    | 2901/5000 [00:18<00:16, 127.62it/s][A[A
    
     58%|█████▊    | 2922/5000 [00:18<00:14, 142.06it/s][A[A
    
     59%|█████▊    | 2937/5000 [00:18<00:15, 131.76it/s][A[A
    
     59%|█████▉    | 2952/5000 [00:18<00:14, 136.64it/s][A[A
    
     59%|█████▉    | 2972/5000 [00:18<00:13, 150.21it/s][A[A
    
     60%|█████▉    | 2990/5000 [00:18<00:12, 157.36it/s][A[A
    
     60%|██████    | 3007/5000 [00:18<00:13, 145.18it/s][A[A
    
     60%|██████    | 3023/5000 [00:18<00:13, 145.16it/s][A[A
    
     61%|██████    | 3041/5000 [00:18<00:12, 152.44it/s][A[A
    
     61%|██████    | 3057/5000 [00:19<00:13, 142.57it/s][A[A
    
     61%|██████▏   | 3072/5000 [00:19<00:13, 142.75it/s][A[A
    
     62%|██████▏   | 3088/5000 [00:19<00:13, 146.11it/s][A[A
    
     62%|██████▏   | 3107/5000 [00:19<00:12, 156.91it/s][A[A
    
     62%|██████▏   | 3124/5000 [00:19<00:12, 151.42it/s][A[A
    
     63%|██████▎   | 3142/5000 [00:19<00:11, 157.55it/s][A[A
    
     63%|██████▎   | 3159/5000 [00:19<00:12, 149.95it/s][A[A
    
     64%|██████▎   | 3175/5000 [00:19<00:12, 142.11it/s][A[A
    
     64%|██████▍   | 3190/5000 [00:20<00:13, 131.90it/s][A[A
    
     64%|██████▍   | 3207/5000 [00:20<00:12, 139.92it/s][A[A
    
     64%|██████▍   | 3222/5000 [00:20<00:12, 140.33it/s][A[A
    
     65%|██████▍   | 3237/5000 [00:20<00:12, 141.18it/s][A[A
    
     65%|██████▌   | 3252/5000 [00:20<00:12, 140.80it/s][A[A
    
     65%|██████▌   | 3267/5000 [00:20<00:12, 141.14it/s][A[A
    
     66%|██████▌   | 3283/5000 [00:20<00:11, 143.80it/s][A[A
    
     66%|██████▌   | 3298/5000 [00:20<00:11, 145.32it/s][A[A
    
     66%|██████▋   | 3316/5000 [00:20<00:11, 152.63it/s][A[A
    
     67%|██████▋   | 3332/5000 [00:20<00:11, 147.37it/s][A[A
    
     67%|██████▋   | 3347/5000 [00:21<00:11, 144.79it/s][A[A
    
     67%|██████▋   | 3362/5000 [00:21<00:12, 131.86it/s][A[A
    
     68%|██████▊   | 3376/5000 [00:21<00:13, 118.92it/s][A[A
    
     68%|██████▊   | 3389/5000 [00:21<00:13, 121.43it/s][A[A
    
     68%|██████▊   | 3402/5000 [00:21<00:13, 121.50it/s][A[A
    
     68%|██████▊   | 3417/5000 [00:21<00:12, 128.21it/s][A[A
    
     69%|██████▊   | 3431/5000 [00:21<00:12, 128.99it/s][A[A
    
     69%|██████▉   | 3445/5000 [00:21<00:12, 123.36it/s][A[A
    
     69%|██████▉   | 3458/5000 [00:22<00:14, 106.81it/s][A[A
    
     69%|██████▉   | 3470/5000 [00:22<00:15, 100.28it/s][A[A
    
     70%|██████▉   | 3485/5000 [00:22<00:13, 110.46it/s][A[A
    
     70%|███████   | 3501/5000 [00:22<00:12, 120.73it/s][A[A
    
     70%|███████   | 3519/5000 [00:22<00:11, 127.27it/s][A[A
    
     71%|███████   | 3536/5000 [00:22<00:10, 134.76it/s][A[A
    
     71%|███████   | 3552/5000 [00:22<00:10, 140.03it/s][A[A
    
     71%|███████▏  | 3570/5000 [00:22<00:09, 149.32it/s][A[A
    
     72%|███████▏  | 3586/5000 [00:22<00:09, 142.78it/s][A[A
    
     72%|███████▏  | 3601/5000 [00:23<00:10, 134.20it/s][A[A
    
     72%|███████▏  | 3615/5000 [00:23<00:11, 124.08it/s][A[A
    
     73%|███████▎  | 3632/5000 [00:23<00:10, 132.94it/s][A[A
    
     73%|███████▎  | 3649/5000 [00:23<00:09, 142.05it/s][A[A
    
     73%|███████▎  | 3664/5000 [00:23<00:09, 142.81it/s][A[A
    
     74%|███████▎  | 3681/5000 [00:23<00:08, 148.78it/s][A[A
    
     74%|███████▍  | 3698/5000 [00:23<00:08, 152.20it/s][A[A
    
     74%|███████▍  | 3716/5000 [00:23<00:08, 158.76it/s][A[A
    
     75%|███████▍  | 3733/5000 [00:24<00:09, 139.53it/s][A[A
    
     75%|███████▍  | 3748/5000 [00:24<00:09, 134.41it/s][A[A
    
     75%|███████▌  | 3766/5000 [00:24<00:08, 145.01it/s][A[A
    
     76%|███████▌  | 3782/5000 [00:24<00:08, 135.59it/s][A[A
    
     76%|███████▌  | 3797/5000 [00:24<00:08, 139.28it/s][A[A
    
     76%|███████▌  | 3812/5000 [00:24<00:08, 141.06it/s][A[A
    
     77%|███████▋  | 3827/5000 [00:24<00:08, 137.71it/s][A[A
    
     77%|███████▋  | 3842/5000 [00:24<00:08, 140.27it/s][A[A
    
     77%|███████▋  | 3861/5000 [00:24<00:07, 149.85it/s][A[A
    
     78%|███████▊  | 3878/5000 [00:25<00:07, 153.38it/s][A[A
    
     78%|███████▊  | 3898/5000 [00:25<00:06, 164.66it/s][A[A
    
     78%|███████▊  | 3915/5000 [00:25<00:07, 143.02it/s][A[A
    
     79%|███████▊  | 3935/5000 [00:25<00:06, 155.89it/s][A[A
    
     79%|███████▉  | 3953/5000 [00:25<00:06, 162.23it/s][A[A
    
     79%|███████▉  | 3970/5000 [00:25<00:06, 160.79it/s][A[A
    
     80%|███████▉  | 3990/5000 [00:25<00:06, 165.00it/s][A[A
    
     80%|████████  | 4007/5000 [00:25<00:06, 165.29it/s][A[A
    
     80%|████████  | 4024/5000 [00:25<00:05, 166.15it/s][A[A
    
     81%|████████  | 4041/5000 [00:25<00:05, 164.44it/s][A[A
    
     81%|████████  | 4058/5000 [00:26<00:06, 148.83it/s][A[A
    
     82%|████████▏ | 4077/5000 [00:26<00:06, 153.01it/s][A[A
    
     82%|████████▏ | 4093/5000 [00:26<00:05, 152.47it/s][A[A
    
     82%|████████▏ | 4113/5000 [00:26<00:05, 163.01it/s][A[A
    
     83%|████████▎ | 4131/5000 [00:26<00:05, 165.46it/s][A[A
    
     83%|████████▎ | 4148/5000 [00:26<00:05, 147.52it/s][A[A
    
     83%|████████▎ | 4164/5000 [00:26<00:05, 139.43it/s][A[A
    
     84%|████████▎ | 4179/5000 [00:26<00:06, 124.39it/s][A[A
    
     84%|████████▍ | 4197/5000 [00:27<00:05, 135.02it/s][A[A
    
     84%|████████▍ | 4212/5000 [00:27<00:05, 138.26it/s][A[A
    
     85%|████████▍ | 4228/5000 [00:27<00:05, 141.68it/s][A[A
    
     85%|████████▍ | 4244/5000 [00:27<00:05, 146.04it/s][A[A
    
     85%|████████▌ | 4259/5000 [00:27<00:05, 143.40it/s][A[A
    
     86%|████████▌ | 4276/5000 [00:27<00:04, 149.19it/s][A[A
    
     86%|████████▌ | 4292/5000 [00:27<00:04, 146.31it/s][A[A
    
     86%|████████▌ | 4309/5000 [00:27<00:04, 152.22it/s][A[A
    
     86%|████████▋ | 4325/5000 [00:27<00:04, 153.25it/s][A[A
    
     87%|████████▋ | 4343/5000 [00:28<00:04, 158.45it/s][A[A
    
     87%|████████▋ | 4359/5000 [00:28<00:04, 144.03it/s][A[A
    
     87%|████████▋ | 4374/5000 [00:28<00:04, 140.35it/s][A[A
    
     88%|████████▊ | 4390/5000 [00:28<00:04, 144.51it/s][A[A
    
     88%|████████▊ | 4405/5000 [00:28<00:04, 134.74it/s][A[A
    
     88%|████████▊ | 4421/5000 [00:28<00:04, 139.22it/s][A[A
    
     89%|████████▊ | 4436/5000 [00:28<00:03, 141.49it/s][A[A
    
     89%|████████▉ | 4451/5000 [00:28<00:03, 139.30it/s][A[A
    
     89%|████████▉ | 4469/5000 [00:28<00:03, 146.80it/s][A[A
    
     90%|████████▉ | 4488/5000 [00:29<00:03, 157.12it/s][A[A
    
     90%|█████████ | 4505/5000 [00:29<00:03, 153.03it/s][A[A
    
     90%|█████████ | 4521/5000 [00:29<00:03, 151.52it/s][A[A
    
     91%|█████████ | 4540/5000 [00:29<00:02, 154.40it/s][A[A
    
     91%|█████████ | 4557/5000 [00:29<00:02, 157.47it/s][A[A
    
     91%|█████████▏| 4573/5000 [00:29<00:03, 138.98it/s][A[A
    
     92%|█████████▏| 4588/5000 [00:29<00:03, 124.63it/s][A[A
    
     92%|█████████▏| 4605/5000 [00:29<00:02, 135.41it/s][A[A
    
     92%|█████████▏| 4622/5000 [00:30<00:02, 142.00it/s][A[A
    
     93%|█████████▎| 4637/5000 [00:30<00:02, 142.77it/s][A[A
    
     93%|█████████▎| 4652/5000 [00:30<00:02, 143.35it/s][A[A
    
     93%|█████████▎| 4671/5000 [00:30<00:02, 154.53it/s][A[A
    
     94%|█████████▎| 4687/5000 [00:30<00:02, 146.55it/s][A[A
    
     94%|█████████▍| 4703/5000 [00:30<00:02, 146.83it/s][A[A
    
     94%|█████████▍| 4720/5000 [00:30<00:01, 151.21it/s][A[A
    
     95%|█████████▍| 4736/5000 [00:30<00:02, 128.67it/s][A[A
    
     95%|█████████▌| 4750/5000 [00:30<00:01, 127.69it/s][A[A
    
     95%|█████████▌| 4764/5000 [00:31<00:01, 124.48it/s][A[A
    
     96%|█████████▌| 4779/5000 [00:31<00:01, 121.88it/s][A[A
    
     96%|█████████▌| 4797/5000 [00:31<00:01, 132.79it/s][A[A
    
     96%|█████████▌| 4812/5000 [00:31<00:01, 137.11it/s][A[A
    
     97%|█████████▋| 4827/5000 [00:31<00:01, 128.43it/s][A[A
    
     97%|█████████▋| 4846/5000 [00:31<00:01, 142.20it/s][A[A
    
     97%|█████████▋| 4864/5000 [00:31<00:00, 150.79it/s][A[A
    
     98%|█████████▊| 4880/5000 [00:31<00:00, 147.02it/s][A[A
    
     98%|█████████▊| 4900/5000 [00:31<00:00, 152.57it/s][A[A
    
     98%|█████████▊| 4916/5000 [00:32<00:00, 143.79it/s][A[A
    
     99%|█████████▊| 4931/5000 [00:32<00:00, 141.16it/s][A[A
    
     99%|█████████▉| 4952/5000 [00:32<00:00, 154.66it/s][A[A
    
     99%|█████████▉| 4969/5000 [00:32<00:00, 146.32it/s][A[A
    
    100%|█████████▉| 4992/5000 [00:32<00:00, 163.56it/s][A[A
    
    100%|██████████| 5000/5000 [00:32<00:00, 153.47it/s]

Notice that we are fitting 13 parameters to 10 points. Ordinarily this would scream overfitting, but thefocus of our parameters is at different levels, and in the hierarchial set up, 10 of these parameters are really pooled together from one sigma. So the effective number of parameters is something lower.



```python
pm.traceplot(trace3)
```





    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x12ca81ac8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x12ddac470>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x12d90d908>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x12d9f47f0>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x12db152e8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x12daa6668>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x12c709940>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x12203fbe0>]], dtype=object)




![png](Islands2_files/Islands2_71_1.png)




```python
np.mean(trace3['diverging'])
```





    0.00020000000000000001





```python
pm.df_summary(trace3)
```





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>sd</th>
      <th>mc_error</th>
      <th>hpd_2.5</th>
      <th>hpd_97.5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>betap</th>
      <td>0.259332</td>
      <td>0.081348</td>
      <td>0.001298</td>
      <td>0.100756</td>
      <td>0.424845</td>
    </tr>
    <tr>
      <th>alpha</th>
      <td>3.442723</td>
      <td>0.120630</td>
      <td>0.001830</td>
      <td>3.195969</td>
      <td>3.677846</td>
    </tr>
    <tr>
      <th>alphasoc__0</th>
      <td>-0.206343</td>
      <td>0.240233</td>
      <td>0.002727</td>
      <td>-0.699141</td>
      <td>0.249360</td>
    </tr>
    <tr>
      <th>alphasoc__1</th>
      <td>0.042643</td>
      <td>0.221406</td>
      <td>0.002891</td>
      <td>-0.400023</td>
      <td>0.488498</td>
    </tr>
    <tr>
      <th>alphasoc__2</th>
      <td>-0.045285</td>
      <td>0.194054</td>
      <td>0.002154</td>
      <td>-0.436802</td>
      <td>0.338603</td>
    </tr>
    <tr>
      <th>alphasoc__3</th>
      <td>0.331428</td>
      <td>0.190349</td>
      <td>0.002922</td>
      <td>-0.043676</td>
      <td>0.695233</td>
    </tr>
    <tr>
      <th>alphasoc__4</th>
      <td>0.045887</td>
      <td>0.176483</td>
      <td>0.002060</td>
      <td>-0.297662</td>
      <td>0.399840</td>
    </tr>
    <tr>
      <th>alphasoc__5</th>
      <td>-0.320044</td>
      <td>0.203813</td>
      <td>0.002330</td>
      <td>-0.733556</td>
      <td>0.053826</td>
    </tr>
    <tr>
      <th>alphasoc__6</th>
      <td>0.147556</td>
      <td>0.171045</td>
      <td>0.002221</td>
      <td>-0.187612</td>
      <td>0.479167</td>
    </tr>
    <tr>
      <th>alphasoc__7</th>
      <td>-0.173431</td>
      <td>0.182775</td>
      <td>0.002013</td>
      <td>-0.543019</td>
      <td>0.185225</td>
    </tr>
    <tr>
      <th>alphasoc__8</th>
      <td>0.277231</td>
      <td>0.174836</td>
      <td>0.002547</td>
      <td>-0.055993</td>
      <td>0.627711</td>
    </tr>
    <tr>
      <th>alphasoc__9</th>
      <td>-0.088570</td>
      <td>0.289709</td>
      <td>0.004807</td>
      <td>-0.673901</td>
      <td>0.484124</td>
    </tr>
    <tr>
      <th>sigmasoc</th>
      <td>0.311862</td>
      <td>0.127298</td>
      <td>0.002352</td>
      <td>0.097031</td>
      <td>0.563365</td>
    </tr>
  </tbody>
</table>
</div>



We can ask the WAIC how many effective parameters it has, and it tells us roughly 5. Thus you really care about the number of hyper-parameters you have, and not so much about the lower level parameters.



```python
pm.waic(trace3, m3c)
```


            log predictive densities exceeds 0.4. This could be indication of
            WAIC starting to fail see http://arxiv.org/abs/1507.04544 for details
            
      """)





    WAIC_r(WAIC=69.536098587001305, WAIC_se=2.4805457558904664, p_WAIC=4.7757394571355052)



We now write code to simulate counterfactuals again, where now we use sampling from $\sigma_{society}$ to simulate our societies. Again, we dont use theano's shareds, opting simply to generate samples for the residual intercepts for multiple societies. How many? As many as the traces. You might have thought you only need to generate as many as there are grid points, ie 30, but at the end the posterior predictive must marginalize over the traces at all these points, and thus marginalizing over the full trace at each point suffices!



```python
def compute_pp2(lpgrid, trace, contact=0):
    alphatrace = trace_or_zero(trace, 'alpha')
    betaptrace = trace_or_zero(trace, 'betap')
    sigmasoctrace = trace_or_zero(trace, 'sigmasoc')
    tl=2*len(trace)
    gl=lpgrid.shape[0]
    lam = np.empty((gl, tl))
    lpgrid = lpgrid - lpgrid.mean()
    #simulate. 5000 alphasocs gen here
    alphasoctrace=np.random.normal(0, sigmasoctrace)
    for i, v in enumerate(lpgrid):
        temp = alphatrace + betaptrace*lpgrid[i] + alphasoctrace
        lam[i,:] = poisson.rvs(np.exp(temp))
    return lam
```




```python
ppsamps = compute_pp2(lpgrid, trace3)

```




```python
ppmed = np.median(ppsamps, axis=1)
pphpd = pm.stats.hpd(ppsamps.T)
plt.plot(df[df['clevel']==1].logpop, df[df['clevel']==1].total_tools,'o', color="g")
plt.plot(df[df['clevel']==0].logpop, df[df['clevel']==0].total_tools,'o', color="r")
plt.plot(lpgrid, ppmed, color="b")
plt.fill_between(lpgrid, pphpd[:,0], pphpd[:,1], color="b", alpha=0.1)
plt.ylim([0, 300])
```





    (0, 300)




![png](Islands2_files/Islands2_79_1.png)


The envelope of predictions is much wider here. This is because of the varying intercepts, and it reflects the fact that there is much more variation in the data than os expected from a pure poisson model.
