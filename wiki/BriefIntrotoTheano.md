---
title: A Brief Intro to Theano
shorttitle: BriefIntrotoTheano
notebook: BriefIntrotoTheano.ipynb
noline: 1
summary: ""
layout: wiki
---
{% assign links = site.data.wikilinks %}

## Introduction to Mathematical Expressions with Theano

Stolen shamelessly from https://github.com/fonnesbeck/Bios8366/blob/master/notebooks/Section4_3-Hamiltonian-Monte-Carlo.ipynb

Theano is a Python library that allows you to define, optimize, and evaluate mathematical expressions involving multi-dimensional arrays efficiently. Theano features:

* __tight integration with numpy__ – Use numpy.ndarray in Theano-compiled functions.
* __transparent use of a GPU__ – Perform data-intensive calculations up to 140x faster than with CPU.(float32 only)
* __efficient symbolic differentiation__ – Theano does your derivatives for function with one or many inputs.
* __speed and stability optimizations__ – Get the right answer for log(1+x) even when x is really tiny.
* __dynamic C code generation__ – Evaluate expressions faster.
* __extensive unit-testing and self-verification__ – Detect and diagnose errors.

Theano is part programming language, part compiler. It is often used to build machine learning, though it is not in itself a machine learning toolkit; think of it as a **mathematical toolkit**.

After a brief introduction to the Theano package, we will use it to implement a modern MCMC algorithm, *Hamiltonian Monte Carlo (HMC)*.

### Installing Theano

The easiest way to install Theano is to build it from source, using **pip**:

```bash
pip install --upgrade --no-deps pip
```

## Adding Two Scalars

To get us started with Theano and get a feel of what we're working with, 
let's make a simple function: add two numbers together. Here is how you do
it:

### Step 1 - Declaring Variables



```python
from theano import function, shared
from theano import tensor as T
import theano

x = T.dscalar('x')
y = T.dscalar('y')
```


In Theano, all symbols must be typed. In particular, `T.dscalar`
is the type we assign to "0-dimensional arrays (`scalar`) of doubles
(`d`)". It is a Theano `type`.



```python
type(x)
```





    theano.tensor.var.TensorVariable





```python
x.type
```





    TensorType(float64, scalar)





```python
T.dscalar
```





    TensorType(float64, scalar)



### Step 2 - Symbolic Expressions

The second step is to combine *x* and *y* into their sum *z*:



```python
z = x + y
```


*z* is yet another *Variable* which represents the addition of
*x* and *y*. You can use the `pp` function to *pretty-print* out the computation associated to *z*.




```python
from theano.printing import pp
print(pp(z))
```


    (x + y)


### Step 3 - Compiling a Function

The last step is to create a function taking *x* and *y* as inputs
and giving *z* as output:



```python
f = function([x, y], z)
```


The first argument to `function()` is a list of Variables
that will be provided as inputs to the function. The second argument
is a single Variable *or* a list of Variables. For either case, the second
argument is what we want to see as output when we apply the function. *f* may
then be used like a normal Python function.


Now we can call the function:



```python
print(f(2, 3))
```


    5.0




```python
print(f(16.3, 12.1))
```


    28.4


If you are following along and typing into an interpreter, you may have
noticed that there was a slight delay in executing the ``function``
instruction. Behind the scenes, *f* was being compiled into C code.

Internally, Theano builds a graph structure composed of interconnected `Variable` nodes, `op` nodes and `apply` nodes. 

An `apply` node represents the application of an `op` to some variables. It is important to draw the difference between the definition of a computation represented by an `op` and its application to some actual data which is represented by the apply node. 

Here is the expression graph corresponding to the addition of `x` and `y`:

![expression graph](images/expression_graph.png)

A *Variable* is the main data structure you work with when
using Theano. By calling `T.dscalar` with a string argument, you create a
`Variable` representing a floating-point scalar quantity with the
given name. If you provide no argument, the symbol will be unnamed. Names
are not required, but they can help debugging.

## Adding Two Matrices

If we want to work with matrices instead of scalars, the only change
from the previous example is that you need to instantiate *x* and
*y* using the matrix Types:



```python
x = T.dmatrix('x')
y = T.dmatrix('y')
z = x + y
f = function([x, y], z)
```


``dmatrix`` is the Type for matrices of doubles. Then we can use
our new function on 2D arrays:




```python
f([[1, 2], [3, 4]], [[10, 20], [30, 40]])
```





    array([[ 11.,  22.],
           [ 33.,  44.]])



The following types are available:

* **byte**: ``bscalar, bvector, bmatrix, brow, bcol, btensor3, btensor4``
* **16-bit integers**: ``wscalar, wvector, wmatrix, wrow, wcol, wtensor3, wtensor4``
* **32-bit integers**: ``iscalar, ivector, imatrix, irow, icol, itensor3, itensor4``
* **64-bit integers**: ``lscalar, lvector, lmatrix, lrow, lcol, ltensor3, ltensor4``
* **float**: ``fscalar, fvector, fmatrix, frow, fcol, ftensor3, ftensor4``
* **double**: ``dscalar, dvector, dmatrix, drow, dcol, dtensor3, dtensor4``
* **complex**: ``cscalar, cvector, cmatrix, crow, ccol, ctensor3, ctensor4``

An example of a slightly more interesting function is the logistic curve:



```python
x = T.dmatrix('x')
```


The logistic transformation:



```python
s = 1 / (1 + T.exp(-x))
```




```python
logistic = function([x], s)
print(logistic([[0, 1], [-1, -2]]))
```


    [[ 0.5         0.73105858]
     [ 0.26894142  0.11920292]]


Theano supports functions with multiple outputs. For example, we can
compute the elementwise difference, absolute difference, and
squared difference between two matrices *a* and *b* at the same time.



```python
a, b = T.dmatrices('a', 'b')
diff = a - b
abs_diff = abs(diff)
diff_squared = diff ** 2
```


When we use the function `f`, it returns the three computed results as a list.



```python
f = function([a, b], [diff, abs_diff, diff_squared])

f([[1, 1], [1, 1]], [[0, 1], [2, 3]])
```





    [array([[ 1.,  0.],
            [-1., -2.]]), array([[ 1.,  0.],
            [ 1.,  2.]]), array([[ 1.,  0.],
            [ 1.,  4.]])]



## Setting a Default Value for an Argument
 
Let's say you want to define a function that adds two numbers, except that if you only provide one number, the other input is assumed to be one. In Python, the default value for parameters achieves this effect.

In Theano we make use of the [In](http://deeplearning.net/software/theano/library/compile/io.html#function-inputs) class, which allows you to specify properties of your function's parameters with greater detail. Here we give a default value of 1 for y by creating an In instance with its value field set to 1. Inputs with default values must follow inputs without default values (like Python's functions). There can be multiple inputs with default values. These parameters can be set positionally or by name, as in standard Python.



```python
from theano import In

x, y, w = T.dscalars('x', 'y', 'w')
z = (x + y) * w
g = function([x, In(y, value=1), In(w, value=2, name='w_by_name')], z)
```




```python
print('g(33) = {}'.format(g(33)))
```


    g(33) = 68.0




```python
print('g(33, 0, 1) = {}'.format(g(33, 0, 1)))
```


    g(33, 0, 1) = 33.0




```python
print('g(33, w_by_name=1) = {}'.format(g(33, w_by_name=1)))
```


    g(33, w_by_name=1) = 34.0




```python
print('g(33, w_by_name=1, y=0) = {}'.format(g(33, w_by_name=1, y=0)))
```


    g(33, w_by_name=1, y=0) = 33.0


### Random Numbers

Because in Theano you first express everything symbolically and afterwards compile this expression to get functions, using pseudo-random numbers is not as straightforward as it is in NumPy.

The way to think about putting randomness into Theano’s computations is to put random variables in your graph. Theano will allocate a NumPy `RandomStream` object (a random number generator) for each such variable, and draw from it as necessary. We will call this sort of sequence of random numbers a random stream. Random streams are at their core shared variables, so the observations on shared variables hold here as well. 



```python
from theano.tensor.shared_randomstreams import RandomStreams

srng = RandomStreams(seed=234)
rv_u = srng.uniform((2,2))
f = function([], rv_u)
```




```python
f()
```


## Looping in Theano

The `scan` function provides the ability to write loops in Theano. We are not able to use Python `for` loops with Theano because Theano needs to be able to build and optimize the expression graph before compiling it into faster code, and be able to use symbolic differentiation for calculating gradients.

### Simple loop with accumulation

Assume that, given $k$ you want to get $A^k$ using a loop. More precisely, if $A$ is a tensor you want to compute $A^k$ elementwise. The python code might look like:

```python
result = 1
for i in range(k):
  result = result * A
```

There are three things here that we need to handle: the initial value assigned to result, the accumulation of results in result, and the unchanging variable A. Unchanging variables are passed to scan as non_sequences. Initialization occurs in outputs_info, and the accumulation happens automatically.

The equivalent Theano code would be:



```python
k = T.iscalar("k")
A = T.vector("A")

# Symbolic description of the result
result, updates = theano.scan(fn=lambda prior_result, A: prior_result * A,
                              outputs_info=T.ones_like(A),
                              non_sequences=A,
                              n_steps=k)

# We only care about A**k, but scan has provided us with A**1 through A**k.
# Discard the values that we don't care about. Scan is smart enough to
# notice this and not waste memory saving them.
final_result = result[-1]

# compiled function that returns A**k
power = theano.function(inputs=[A,k], outputs=final_result, updates=updates)

print(power(range(10),2))
print(power(range(10),4))
```


Let us go through the example line by line. What we did is first to **construct a function** (using a lambda expression) that given `prior_result` and `A` returns `prior_result * A`. The order of parameters is fixed by `scan`: the output of the prior call to `fn` is the first parameter, followed by all non-sequences.

Next we **initialize the output** as a tensor with same shape and `dtype` as `A`, filled with ones. We give `A` to `scan` as a non sequence parameter and specify the number of steps `k` to iterate over our `lambda` expression.

Scan **returns a tuple** containing our result (`result`) and a dictionary of updates (empty in this case). Note that the result is not a matrix, but a 3D tensor containing the value of $A^k$ for each step. We want the last value (after k steps) so we compile a function to return just that. Note that there is an optimization, that at compile time will detect that you are using just the last value of the result and ensure that scan does not store all the intermediate values that are used. So do not worry if `A` and `k` are large.

In addition to looping a fixed number of times, scan can iterate over the leading dimension of tensors (similar to Python’s `for x in a_list`).

The tensor(s) to be looped over should be provided to `scan` using the `sequences` keyword argument.
