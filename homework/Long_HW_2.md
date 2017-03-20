---
title: Long Homework 2
shorttitle: Long Homework 2
notebook: Long_HW_2.ipynb
noline: 1
layout: wiki
---

## AMPTH 207: Stochastic Methods for Data Analysis, Inference and Optimization

##### Data: noisy_image.pickle

**Due Date:** Thursday, March 23rd, 2017 at 11:59pm

**Instructions:**

- Upload your final answers as well as your iPython notebook containing all work to Canvas.

- Structure your notebook and your work to maximize readability.

## Problem 1. The Ising Model with Metropolis and Gibbs

The true power of the Monte Carlo method becomes apparent when we deal with integrals over many dimensions, such as those often found in statistical physics. In this exercise we consider the 2-D Ising model, which is a very idealized representation of a ferromagnetic metal. The model consists of an $L\times L$ lattice of spins  $s_{ij}$ (with $i, j = 1, \ldots , L$) each of which can take only two values, up ($s_{ij}$ = 1) and down ($s_{ij}$ = -1). The total energy for the system is taken to be:

$$
E = -\sum_{ij} \frac{1}{2} s_{ij}  \left[ s_{\rm{up}(ij)} + s_{\rm{down}(ij)} + s_{\rm{left}(ij)} + s_{\rm{right}(ij)} \right],  
$$

where $s_{\rm{up}(ij)}$ denotes the spin immediately above $s_{ij}$ in the grid, $s_{\rm{right}(ij)}$ the spin immediately to its right, and so on. We see that configurations in which nearby spins are aligned are energetically preferred. When the system is in thermal equilibrium, at the temperature $T$, the probability of finding it in a given configuration $\{s\}=(s_{11}, s_{12}, \ldots)$ of the spins is given by the Boltzmann factor,

$$W(\{ {s} \}) = Z^{-1} e^{ -\beta E[ \{ s \} ] }, $$

where $Z$ is given by the sum of the exponential over all possible spin configurations, and where $\beta$ is the inverse temperature (measured in units of the Boltzmann constant $k_B$). The thermal equilibrium values of all the macroscopic quantities for the system (such as the magnetization, $M(\{s\}) =\sum_{ij} s_{ij}$ are found by averaging over all spin configurations, weighted by $W$. For instance,

$$
\left< M \right> := \sum_{\{s\}} W(\{ {s} \})M(\{s\})
$$ 

Thus:

$$
\left< M \right> = \sum_{s_{11}=\pm 1} \sum_{s_{12}=\pm 1} \ldots \sum_{s_{LL}=\pm 1} \left[ W(\{s\}) \times \sum_{ij} s_{ij} \right],
$$
Computing thermal equilibrium values using the above equation is effectively out of the reach of standard techniques, but not so if we use Monte Carlo.

### Sampling: Metropolis

The Metropolis algorithm generates a sequence of configurations that have approximately the required probability distribution. This is how it works:

* We start with a completely random spin configuration $\{s \}$ (each spin chosen to be up or down with 50% probability).

* To generate the next configuration in the sequence, we select a point $(i, j$) in the lattice, and we compute the energy change $\Delta E$ by flipping the corresponding spin, $s_{ij}$,

$$
\Delta E = 2s_{ij}  \left[ s_{\rm{up}(ij)} + s_{\rm{down}(ij)} + s_{\rm{left}(ij)} + s_{\rm{right}(ij)} \right]  
$$

* if E is zero or negative, we accept the flip (i.e., we always accept changes that lower the total energy). If E is positive, we accept the flip with probability 

$$P (\Delta E) = \exp(-\beta \Delta E).$$ 

We repeat this Metropolis step L $\times$ L times, once for each location $(i, j)$ in the lattice (the repeated Metropolis steps constitute a Metropolis sweep). The updated points are used as they become available.

### Sampling: Gibbs
Since the spins are correlated with their neighbors, it is perhaps more natural to adopt a Gibbs approach. If we select a node $s_{ij}$ in the lattice, it is easy to show using the equations above that the full conditional probability of it having a spin of 1 given all the other spins is:

$$
P(s_{ij}=1\vert\textrm{all other spins}) = \frac{1}{1+\exp\left(-2\beta(s_{\rm{up}(ij)} + s_{\rm{down}(ij)} + s_{\rm{left}(ij)} + s_{\rm{right}(ij)})\right)}
$$

So the Gibbs model is to loop over each spin in the lattice, and with the above probability set the current spin to 1, otherwise setting it to −1.

### Monte Carlo
Regardless of the sampling method, the Monte Carlo estimate of the ensemble average of a macroscopic quantity (say M) is then

$$ \left< M \right> = \frac{1}{N} \sum_{k=1}^{N} \sum_{ij} s_{ij}^{(k)} $$

where $ s_{ij}^{(k)}$ is the sequence of the spin configurations obtained as usual by evolving the Markov Chain from the initial configuration. And as usual, its best to omit a certain number of configurations at the beginning of the sequence; these are not distributed with the proper probabilities, because the system has not yet reached thermal equilibrium (stationarity). How long the system takes to thermalize depends on several factors, and the length of the required warm-up period is usually estimated empirically.

### Part A
- Make the assumption of periodic boundary conditions. This means that the right neighbor of last column are the corresponding points in the first column; and the bottom neighbors in the last row are the corresponding points in the first row (picture your lattice as a grid on a flat sheet of paper, where neighbouring points are connected by lines; then draw lines connecting corresponding points on the left and right edges of your sheet; also draw lines connecting corresponding points on the top and bottom edges). 

  Write a program that generates a Gibbs sequence of spin configurations for the  $100 \times 100$ L 2-D Ising model. Visualize the configurations on the screen as they are computed.

- Use the Gibbs estimates to evaluate the energy per particle and magnetization per particle of the 2-D Ising model as a function of temperature, and plot these functions. At some value of $\beta$ you will observe a sudden change in energy and magnetization, called a phase transition. Where does this occur? 

***Hint:*** *first plot the magnetizations and energies as a function of the iteration to make sure they have stabilized*.


### Part B

Now add an external field of magnitude $h > 0$, which can be interpreted as an energy term added to each spin, and repeat the experiment in Part (A).

In this case:

$$
E = -\sum_{ij} \frac{1}{2} s_{ij}  \left[ s_{\rm{up}(ij)} + s_{\rm{down}(ij)} + s_{\rm{left}(ij)} + s_{\rm{right}(ij)} \right]  - \sum_{ij} h_{ij} s_{ij}
$$

where $h_{ij}$ is the magnitude of the external field in the $(i,j)$ position of the lattice, and $s_{ij}$ the spin in that location in the lattice. For the sake of simplicity, we will assume that $h$ is constant over the entire lattice. 

Note that the above implies that adjacent spins with the same sign result in lower energy than adjacent spins with opposite signs. Also, if the spin direction and the direction of the external magnetic field are aligned, the energy is reduced.

Use a low temperature, $\beta=1$, and plot the magnetization as a function of the Gibbs itearation for the following values of the external field: $h = -1.0,-0.1,-0.05,0.05,0.1,1.0$. Discuss the effect of adding this external field on the magnetization properties.

## Problem 2. Image de-noising

Suppose we observe a "noisy" image $Y \in \{-1,1\}^L$ (a lattice) based on a "noise-free" image $Z \in \{-1,1\}^L$. We will denote the random variable representing the value at the $i$-pixel in the two images by $Y_i$ and $Z_i$, respectively. A plausible choice for modelling the noise in $Y$ is by using independent bit-flips:


$$
P(Y_i=y_i\vert Z_i=z_i) = P(Z_i=z_i\vert Y_i=y_i) = \left\{
        \begin{array}{ll}
            q & \quad y_i = z_i \\
            1-q & \quad y_i \neq z_i
        \end{array}
    \right.
$$

where $q >0$. Simply put, if the $i$-th pixel $Z_i$ in the noise-free image is a -1, then the $i$-th pixel in $Y$ will be wrong (i.e. $Y_i=1$) with probability $1 - q$.

We want to recover $Z$, the original noise-free image. The problem is hard unless we make an assumption about $Z$. We will assume that the image is "clumpy", i.e., that neighboring pixels are correlated. We can translate this assumption directly into the prior probability of $Z$:

$$
\text{Prior on Lattice }Z = P(Z) \propto \exp \left( \sum_{i,j}z_iz_j\right)
$$

Looks familiar? Yes, that is an Ising model with no external field! Now, applying Bayes' rule, and setting $$h=\log\left(\frac{q}{1-q}\right) = \mathrm{logit}(q),$$ 
we can write the posterior probability as:

$$
\log P(Z\vert Y) \propto \exp \left( \sum_{i,j}z_iz_j + \sum_i h\cdot z_i\cdot y_i \right)
$$

This is exactly an Ising model with a varying external field $h\cdot y_i$. We only need to set $\beta=2$ and halve $h$.

### Part A

Using a Gibbs sampler, de-noise the image contained in the file "noisy_image.pickle" and plot your result. How many Gibbs iterations do you need in order to obtain a clean image?
