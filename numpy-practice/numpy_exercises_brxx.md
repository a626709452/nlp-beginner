
# 100 numpy exercises

This is a collection of exercises that have been collected in the numpy mailing list, on stack overflow and in the numpy documentation. The goal of this collection is to offer a quick reference for both old and new users but also to provide a set of exercices for those who teach.


If you find an error or think you've a better way to solve some of them, feel free to open an issue at <https://github.com/rougier/numpy-100>

#### 1. Import the numpy package under the name `np` (★☆☆)


```python
import numpy as np
```

#### 2. Print the numpy version and the configuration (★☆☆)


```python
print np.__version__
print np.version.version
np.show_config()
```

    1.13.1
    1.13.1
    lapack_opt_info:
        libraries = ['openblas', 'openblas']
        library_dirs = ['/usr/local/lib']
        define_macros = [('HAVE_CBLAS', None)]
        language = c
    blas_opt_info:
        libraries = ['openblas', 'openblas']
        library_dirs = ['/usr/local/lib']
        define_macros = [('HAVE_CBLAS', None)]
        language = c
    openblas_info:
        libraries = ['openblas', 'openblas']
        library_dirs = ['/usr/local/lib']
        define_macros = [('HAVE_CBLAS', None)]
        language = c
    blis_info:
      NOT AVAILABLE
    openblas_lapack_info:
        libraries = ['openblas', 'openblas']
        library_dirs = ['/usr/local/lib']
        define_macros = [('HAVE_CBLAS', None)]
        language = c
    lapack_mkl_info:
      NOT AVAILABLE
    blas_mkl_info:
      NOT AVAILABLE
    

#### 3. Create a null vector of size 10 (★☆☆)


```python
a = np.zeros(10)
```

#### 4.  How to find the memory size of any array (★☆☆)


```python
print a.size
print a.itemsize
print a.size * a.itemsize
```

    10
    8
    80
    

#### 5.  How to get the documentation of the numpy add function from the command line? (★☆☆)


```python
!python -c "import numpy;numpy.info(numpy.add)"
# ! for command line
# python -c "" for python command line
```

    add(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])
    
    Add arguments element-wise.
    
    Parameters
    ----------
    x1, x2 : array_like
        The arrays to be added.  If ``x1.shape != x2.shape``, they must be
        broadcastable to a common shape (which may be the shape of one or
        the other).
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or `None`,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        Values of True indicate to calculate the ufunc at that position, values
        of False indicate to leave the value in the output alone.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.
    
    Returns
    -------
    add : ndarray or scalar
        The sum of `x1` and `x2`, element-wise.  Returns a scalar if
        both  `x1` and `x2` are scalars.
    
    Notes
    -----
    Equivalent to `x1` + `x2` in terms of array broadcasting.
    
    Examples
    --------
    >>> np.add(1.0, 4.0)
    5.0
    >>> x1 = np.arange(9.0).reshape((3, 3))
    >>> x2 = np.arange(3.0)
    >>> np.add(x1, x2)
    array([[  0.,   2.,   4.],
           [  3.,   5.,   7.],
           [  6.,   8.,  10.]])


#### 6.  Create a null vector of size 10 but the fifth value which is 1 (★☆☆)


```python
b = np.zeros(10)
b[4] = 1
print b
```

    [ 0.  0.  0.  0.  1.  0.  0.  0.  0.  0.]
    

#### 7.  Create a vector with values ranging from 10 to 49 (★☆☆)


```python
c = np.arange(10, 49)
print c
```

    [10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34
     35 36 37 38 39 40 41 42 43 44 45 46 47 48]
    

#### 8.  Reverse a vector (first element becomes last) (★☆☆)


```python
d = np.arange(10, 49)
print d[::-1]
```

    [48 47 46 45 44 43 42 41 40 39 38 37 36 35 34 33 32 31 30 29 28 27 26 25 24
     23 22 21 20 19 18 17 16 15 14 13 12 11 10]
    

#### 9.  Create a 3x3 matrix with values ranging from 0 to 8 (★☆☆)


```python
print np.arange(0, 9).reshape(3, 3)
```

    [[0 1 2]
     [3 4 5]
     [6 7 8]]
    

#### 10. Find indices of non-zero elements from \[1,2,0,0,4,0\] (★☆☆)


```python
a = np.array([1,2,0,0,4,0])
print np.arange(0, a.size)[a!=0]
# nonzero, return tuple: the first element means the axis=0 of the indice and the second element means the axis=1 of the indice
# eg. if the a[0,0], a[1,2], a[3,4] is nonzero, then the function returns (array([0,1,3]), array(0,2,4)) 
print np.nonzero([1,2,0,0,4,0])[0]
```

    [0 1 4]
    [0 1 4]
    

#### 11. Create a 3x3 identity matrix (★☆☆)


```python
np.eye(3,3)
```




    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])



#### 12. Create a 3x3x3 array with random values (★☆☆)


```python
np.random.random([3,3,3])
```




    array([[[ 0.1149687 ,  0.11742324,  0.0496879 ],
            [ 0.4679223 ,  0.85326528,  0.59758272],
            [ 0.43568485,  0.59190114,  0.1469134 ]],
    
           [[ 0.28930262,  0.20751857,  0.48507756],
            [ 0.17897405,  0.30439575,  0.87228101],
            [ 0.93109626,  0.89743418,  0.40548233]],
    
           [[ 0.32460542,  0.4026128 ,  0.00549765],
            [ 0.57700301,  0.56520973,  0.00621552],
            [ 0.5948916 ,  0.69131941,  0.85273974]]])



#### 13. Create a 10x10 array with random values and find the minimum and maximum values (★☆☆)


```python
a = np.random.random([10,10])
print a
print a.min(), a.max()
```

    [[ 0.80032213  0.72498956  0.85361131  0.78512129  0.49929226  0.81688721
       0.44602114  0.76155528  0.87509068  0.95641327]
     [ 0.56191298  0.65111071  0.86478143  0.75715028  0.8611985   0.63240509
       0.49801171  0.92057144  0.74798693  0.360567  ]
     [ 0.22277218  0.07062793  0.07138634  0.04357424  0.36363091  0.90113656
       0.15396669  0.73890168  0.29390106  0.05413523]
     [ 0.79991803  0.76540378  0.81904112  0.4323953   0.04767712  0.63256814
       0.00936706  0.60683642  0.25948146  0.45985726]
     [ 0.07759964  0.33816513  0.11253799  0.41572803  0.12516572  0.87137914
       0.58804937  0.70377555  0.12044448  0.52083715]
     [ 0.64762716  0.32211405  0.82408713  0.45351712  0.28366987  0.19808923
       0.94414112  0.60254607  0.69578657  0.93088825]
     [ 0.26135163  0.1906105   0.95549539  0.54545559  0.91917358  0.71481284
       0.72722953  0.9934477   0.87595189  0.5727191 ]
     [ 0.21697159  0.41580733  0.86232628  0.41038235  0.59029606  0.26352727
       0.35867328  0.74696697  0.78531885  0.29389729]
     [ 0.38236391  0.25881808  0.8435116   0.35528224  0.56733233  0.71346047
       0.16158264  0.31437791  0.29948883  0.3215291 ]
     [ 0.66431238  0.77748524  0.60988993  0.38382586  0.92164632  0.49754483
       0.65505955  0.28005129  0.24087972  0.67479189]]
    0.00936706267213 0.993447697082
    

#### 14. Create a random vector of size 30 and find the mean value (★☆☆)


```python
a = np.random.random(30)
print a.mean()
```

    0.443143269566
    

#### 15. Create a 2d array with 1 on the border and 0 inside (★☆☆)


```python
a = np.zeros([10,10])
a[[0,-1],:] = 1
a[:,[0,-1]] = 1
print a
```

    [[ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
     [ 1.  0.  0.  0.  0.  0.  0.  0.  0.  1.]
     [ 1.  0.  0.  0.  0.  0.  0.  0.  0.  1.]
     [ 1.  0.  0.  0.  0.  0.  0.  0.  0.  1.]
     [ 1.  0.  0.  0.  0.  0.  0.  0.  0.  1.]
     [ 1.  0.  0.  0.  0.  0.  0.  0.  0.  1.]
     [ 1.  0.  0.  0.  0.  0.  0.  0.  0.  1.]
     [ 1.  0.  0.  0.  0.  0.  0.  0.  0.  1.]
     [ 1.  0.  0.  0.  0.  0.  0.  0.  0.  1.]
     [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]]
    

#### 16. How to add a border (filled with 0's) around an existing array? (★☆☆)


```python
a = np.random.random([3,3])
# print np.pad(a, pad_width=((3,2),(1,4)), mode='maximum')
# 3 on the top and 2 on the bottom, 1 on the left and 4 on the right
print np.pad(a,pad_width=1,mode='constant',constant_values=0)
```

    [[ 0.          0.          0.          0.          0.        ]
     [ 0.          0.55407892  0.29251313  0.82219663  0.        ]
     [ 0.          0.16926169  0.88591793  0.73104758  0.        ]
     [ 0.          0.99875408  0.95801145  0.91474817  0.        ]
     [ 0.          0.          0.          0.          0.        ]]
    

#### 17. What is the result of the following expression? (★☆☆)

```python
0 * np.nan
np.nan == np.nan
np.inf > np.nan
np.nan - np.nan
0.3 == 3 * 0.1
```


```python
# any number operated with nan will get nan
# nan can not be compared
print 0 * np.nan # nan
print np.nan == np.nan # False
print np.inf > np.nan # False
print np.nan - np.nan # nan
print 0.3 == 3 * 0.1 # False
```

    nan
    False
    False
    nan
    False
    

#### 18. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal (★☆☆)


```python
a = np.zeros([5,5])
a[1,0]=1
a[2,1]=2
a[3,2]=3
a[4,3]=4
print a
#k=0对角线，k>0对角线上面，k<0对角线下面
# k = 0 is the diag, k > 0 means above the diag by the offset
b = np.diag(np.arange(1,5), k=-1)
print b
```

    [[ 0.  0.  0.  0.  0.]
     [ 1.  0.  0.  0.  0.]
     [ 0.  2.  0.  0.  0.]
     [ 0.  0.  3.  0.  0.]
     [ 0.  0.  0.  4.  0.]]
    [[0 0 0 0 0]
     [1 0 0 0 0]
     [0 2 0 0 0]
     [0 0 3 0 0]
     [0 0 0 4 0]]
    

#### 19. Create a 8x8 matrix and fill it with a checkerboard pattern (★☆☆)


```python
a = np.zeros([8,8])
a[0::2,1::2] = 1
a[1::2, 0::2] = 1
print a
```

    [[ 0.  1.  0.  1.  0.  1.  0.  1.]
     [ 1.  0.  1.  0.  1.  0.  1.  0.]
     [ 0.  1.  0.  1.  0.  1.  0.  1.]
     [ 1.  0.  1.  0.  1.  0.  1.  0.]
     [ 0.  1.  0.  1.  0.  1.  0.  1.]
     [ 1.  0.  1.  0.  1.  0.  1.  0.]
     [ 0.  1.  0.  1.  0.  1.  0.  1.]
     [ 1.  0.  1.  0.  1.  0.  1.  0.]]
    

#### 20. Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element?


```python
a = np.arange(0,6*7*8).reshape([6,7,8])
print np.nonzero(a==99) # a[1,5,3]
```

    (array([1]), array([5]), array([3]))
    

#### 21. Create a checkerboard 8x8 matrix using the tile function (★☆☆)


```python
# tile function refers to https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.tile.html
a = [[0,1],[1,0]]
print np.tile(a, [4,4])
```

    [[0 1 0 1 0 1 0 1]
     [1 0 1 0 1 0 1 0]
     [0 1 0 1 0 1 0 1]
     [1 0 1 0 1 0 1 0]
     [0 1 0 1 0 1 0 1]
     [1 0 1 0 1 0 1 0]
     [0 1 0 1 0 1 0 1]
     [1 0 1 0 1 0 1 0]]
    

#### 22. Normalize a 5x5 random matrix (★☆☆)


```python
a = np.random.random([5,5])
mean = a.mean()
var = b.std()
a = (a - mean) / var
print a
# vector=(vector-vector.min())/(vector.max()-vector.min())
```

    [[ 0.12434585 -0.50885424 -0.32874708  0.12965104  0.14276829]
     [-0.01081289 -0.14863788 -0.38014819  0.09420089 -0.28363426]
     [ 0.18371215  0.2488285   0.3561165  -0.30227574  0.16760656]
     [-0.34259939  0.39417691 -0.22790312  0.08062595 -0.13458016]
     [ 0.00626271  0.31499683  0.14968638  0.07177528  0.20343911]]
    

#### 23. Create a custom dtype that describes a color as four unsigned bytes (RGBA) (★☆☆)


```python
# dtype refers to https://docs.scipy.org/doc/numpy/reference/generated/numpy.dtype.html
# the difference between astype and dtype can refer to http://www.mamicode.com/info-detail-1180317.html
color = np.dtype([('R',np.ubyte), ('G', np.ubyte), ('B', np.ubyte), ('A', np.ubyte)])
print color
```

    [('R', 'u1'), ('G', 'u1'), ('B', 'u1'), ('A', 'u1')]
    

#### 24. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product) (★☆☆)


```python
a = np.random.random([5, 3])
b = np.random.random([3, 2])
print np.dot(a, b)
```

    [[ 0.26300364  0.41006981]
     [ 0.41434895  0.48906535]
     [ 0.81901843  1.43756999]
     [ 1.12223858  1.67969553]
     [ 0.46690095  0.70416466]]
    

#### 25. Given a 1D array, negate all elements which are between 3 and 8, in place. (★☆☆)


```python
a = np.arange(1, 10)
a[(a > 3) & (a < 8)] *= -1
print a
```

    [ 1  2  3 -4 -5 -6 -7  8  9]
    

#### 26. What is the output of the following script? (★☆☆)

```python
# Author: Jake VanderPlas

print(sum(range(5),-1))
from numpy import *
print(sum(range(5),-1))
```


```python
print(sum(range(5),-1)) # 9,sum is the build-in function in python，sum(sequence[,start]) equal to start+1+2+3+4=9
print(np.sum(range(5),-1)) #numpy.sum(a, axis=None)
```

    9
    10
    

#### 27. Consider an integer vector Z, which of these expressions are legal? (★☆☆)

```python
Z**Z
2 << Z >> 2
Z <- Z
1j*Z
Z/1/1
Z<Z>Z
```


```python
Z = np.arange(0,5)
print Z
print Z ** Z # [0^0, 1^1, 2^2, 3^3, 4^4]
print 2 << Z >> 2
print Z <- Z # -> refer to the return type of the function
print 1j*Z
print Z/1/1
# print Z<Z>Z
```

    [0 1 2 3 4]
    [  1   1   4  27 256]
    [0 1 2 4 8]
    [False False False False False]
    [ 0.+0.j  0.+1.j  0.+2.j  0.+3.j  0.+4.j]
    [0 1 2 3 4]
    

#### 28. What are the result of the following expressions?

```python
np.array(0) / np.array(0)
np.array(0) // np.array(0)
np.array([np.nan]).astype(int).astype(float)
```


```python
from __future__ import division
# print np.array(0) / np.array(0) # nan, runtime warning (float division)
# print np.array(0) // np.array(0) #runtime warning (int division)
np.array([np.nan]).astype(int).astype(float)
```




    array([ -9.22337204e+18])



#### 29. How to round away from zero a float array ? (★☆☆)


```python
a = np.random.uniform(-10, 10, 10)
print a
# copysign for copysign from Z
print (np.copysign(np.ceil(np.abs(a)), a))
```

    [-0.63639402  7.99788506  3.70142702  1.70224684  6.2089118  -6.78357125
      3.48732796 -0.15504115 -6.82567455 -2.6627188 ]
    [-1.  8.  4.  2.  7. -7.  4. -1. -7. -3.]
    

#### 30. How to find common values between two arrays? (★☆☆)


```python
a = np.random.randint(0, 10, 10)
b = np.random.randint(0, 10, 10)
print a
print b
print(np.intersect1d(a, b))
```

    [6 2 8 6 8 3 8 7 1 2]
    [0 7 9 2 7 7 9 0 2 9]
    [2 7]
    

#### 31. How to ignore all numpy warnings (not recommended)? (★☆☆)


```python
# Suicide mode on
defaults = np.seterr(all="ignore")
Z = np.ones(1) / 0

# Back to sanity
_ = np.seterr(**defaults)

#An equivalent way, with a context manager:
with np.errstate(divide='ignore'):
    Z = np.ones(1) / 0
```

#### 32. Is the following expressions true? (★☆☆)

```python
np.sqrt(-1) == np.emath.sqrt(-1)
```


```python
# print np.sqrt(-1) # nan
print np.emath.sqrt(-1) # 1j
```

    1j
    

#### 33. How to get the dates of yesterday, today and tomorrow? (★☆☆)


```python
print np.datetime64('today', 'D') - np.timedelta64(1, 'D')
print np.datetime64('today', 'D')
print np.datetime64('today', 'D') + np.timedelta64(1, 'D')
```

    2017-09-28
    2017-09-29
    2017-09-30
    

#### 34. How to get all the dates corresponding to the month of July 2016? (★★☆)


```python
print np.arange('2016-07', '2016-08', dtype='datetime64[D]')
```

    ['2016-07-01' '2016-07-02' '2016-07-03' '2016-07-04' '2016-07-05'
     '2016-07-06' '2016-07-07' '2016-07-08' '2016-07-09' '2016-07-10'
     '2016-07-11' '2016-07-12' '2016-07-13' '2016-07-14' '2016-07-15'
     '2016-07-16' '2016-07-17' '2016-07-18' '2016-07-19' '2016-07-20'
     '2016-07-21' '2016-07-22' '2016-07-23' '2016-07-24' '2016-07-25'
     '2016-07-26' '2016-07-27' '2016-07-28' '2016-07-29' '2016-07-30'
     '2016-07-31']
    

#### 35. How to compute ((A+B)\*(-A/2)) in place (without copy)? (★★☆)


```python
A = np.ones(3) * 3 
B = np.ones(3) * 3 
temp = ((A+B)*(-A/2))
print A
print B
np.add(A, B, out = A)
np.divide(A, 2, out = A)
np.negative(A, out = A)
np.multiply(A, B, out = A)
print A
print B
print A == temp
```

    [ 3.  3.  3.]
    [ 3.  3.  3.]
    [-9. -9. -9.]
    [ 3.  3.  3.]
    [ True  True  True]
    

#### 36. Extract the integer part of a random array using 5 different methods (★★☆)


```python
a = np.random.uniform(0, 10, 10)
print a
print a.astype(int)
print np.floor(a)
print np.ceil(a) - 1
print a - a % 1
print np.trunc(a) # tail-cut
```

    [ 4.62704819  7.86623433  5.28836127  3.39909293  2.73788321  9.77696496
      7.76151243  2.23373681  4.32456907  8.15502965]
    [4 7 5 3 2 9 7 2 4 8]
    [ 4.  7.  5.  3.  2.  9.  7.  2.  4.  8.]
    [ 4.  7.  5.  3.  2.  9.  7.  2.  4.  8.]
    [ 4.  7.  5.  3.  2.  9.  7.  2.  4.  8.]
    [ 4.  7.  5.  3.  2.  9.  7.  2.  4.  8.]
    

#### 37. Create a 5x5 matrix with row values ranging from 0 to 4 (★★☆)


```python
Z = np.zeros((5, 5))
Z += np.arange(5)
print(Z)
a = np.arange(5)
print np.tile(a, [5,1])
```

    [[ 0.  1.  2.  3.  4.]
     [ 0.  1.  2.  3.  4.]
     [ 0.  1.  2.  3.  4.]
     [ 0.  1.  2.  3.  4.]
     [ 0.  1.  2.  3.  4.]]
    [[0 1 2 3 4]
     [0 1 2 3 4]
     [0 1 2 3 4]
     [0 1 2 3 4]
     [0 1 2 3 4]]
    

#### 38. Consider a generator function that generates 10 integers and use it to build an array (★☆☆)


```python
def generator():
    for x in range(0, 10):
        yield x
# numpy.fromiter ---- build an ndarray object from an iterable object
# numpy.fromiter(iterable, dtype, count = -1)
# count : int, optional The number of items to read from iterable. The default is -1, which means all data is read.
Z = np.fromiter(generator(),dtype=int,count=-1)
print Z
```

    [0 1 2 3 4 5 6 7 8 9]
    

#### 39. Create a vector of size 10 with values ranging from 0 to 1, both excluded (★★☆)


```python
print np.linspace(0, 1, 11, endpoint=False)[1:]
# endpoint : bool, optional If True, stop is the last sample. Otherwise, it is not included. Default is True.
```

    [ 0.09090909  0.18181818  0.27272727  0.36363636  0.45454545  0.54545455
      0.63636364  0.72727273  0.81818182  0.90909091]
    

#### 40. Create a random vector of size 10 and sort it (★★☆)


```python
a = np.random.random(10)
print a
a.sort()
print a
```

    [ 0.41503028  0.04598068  0.23864444  0.5188116   0.40956137  0.14843841
      0.81101092  0.11588231  0.25610178  0.71524119]
    [ 0.04598068  0.11588231  0.14843841  0.23864444  0.25610178  0.40956137
      0.41503028  0.5188116   0.71524119  0.81101092]
    

#### 41. How to sum a small array faster than np.sum? (★★☆)


```python
a = np.random.random(10)
print a
print np.add.reduce(a)
```

    [ 0.6998322   0.12745449  0.89493678  0.0321568   0.89811057  0.51672659
      0.5010029   0.92308019  0.58383015  0.47108241]
    5.64821307895
    

#### 42. Consider two random array A and B, check if they are equal (★★☆)


```python
A = np.random.random(10)
B = np.random.random(10)
# True if two arrays have the same shape and elements, False otherwise.
print np.array_equal(A,B)
# Returns True if two arrays are element-wise equal within a tolerance.
print np.allclose(A, B)
```

    False
    False
    

#### 43. Make an array immutable (read-only) (★★☆)


```python
a = np.eye(3)
print a
print a.flags
a.setflags(write=0)
```

    [[ 1.  0.  0.]
     [ 0.  1.  0.]
     [ 0.  0.  1.]]
      C_CONTIGUOUS : True
      F_CONTIGUOUS : False
      OWNDATA : True
      WRITEABLE : True
      ALIGNED : True
      UPDATEIFCOPY : False
    

#### 44. Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates (★★☆)


```python
a = np.random.randint(1,100,10*2).reshape((10,2))
print a
b = np.arctan2(a[:,0], a[:,1])
c = np.sqrt(a[:,0]**2 + a[:,1]**2)
print zip(b, c)
```

    [[41 13]
     [16 35]
     [53 14]
     [35 85]
     [55 71]
     [81 12]
     [24 13]
     [60 28]
     [82  1]
     [94  6]]
    [(1.2637505947761059, 43.011626335213137), (0.4287780274460164, 38.483762809787713), (1.3125441069838579, 54.817880294662984), (0.39060704369768678, 91.923881554251182), (0.65909004633420809, 89.810912477270818), (1.423717971406494, 81.884064383737083), (1.0743735733900148, 27.294688127912362), (1.1341691669813554, 66.211781428987393), (1.5586018093466447, 82.006097334283623), (1.5070530142622096, 94.19129471453293)]
    

#### 45. Create random vector of size 10 and replace the maximum value by 0 (★★☆)


```python
a = np.random.randint(1,100,10)
print a
a[a==a.max()] = 0
print a
a[a.argmax()]=0
print a
```

    [ 4 75 24 62 87 83 94 25 23 86]
    [ 4 75 24 62 87 83  0 25 23 86]
    [ 4 75 24 62  0 83  0 25 23 86]
    

#### 46. Create a structured array with `x` and `y` coordinates covering the \[0,1\]x\[0,1\] area (★★☆)


```python
# Return coordinate matrices from coordinate vectors.
print np.meshgrid([0,1],[0,1])
```

    [array([[0, 1],
           [0, 1]]), array([[0, 0],
           [1, 1]])]
    

####  47. Given two arrays, X and Y, construct the Cauchy matrix C (Cij =1/(xi - yj))


```python
X = np.random.randint(1,100,3)
Y = np.random.randint(1,100,5)
print X
print Y
from __future__ import division
print 1 / np.subtract.outer(X,Y)
```

    [53 27  7]
    [17 69 86 50 39]
    [[ 0.02777778 -0.0625     -0.03030303  0.33333333  0.07142857]
     [ 0.1        -0.02380952 -0.01694915 -0.04347826 -0.08333333]
     [-0.1        -0.01612903 -0.01265823 -0.02325581 -0.03125   ]]
    

#### 48. Print the minimum and maximum representable value for each numpy scalar type (★★☆)


```python
# iinfo: Machine limits for integer types.
# finfo: Machine limits for float point types.
for dtype in [np.int8, np.int16, np.int32, np.int64]:
    print dtype, ":"
    print np.iinfo(dtype).min
    print np.iinfo(dtype).max
for dtype in [np.float16, np.float32, np.float64, np.float128]:
    print dtype , ":"
    print np.finfo(dtype).min
    print np.finfo(dtype).max
    print np.finfo(dtype).eps
```

    <type 'numpy.int8'> :
    -128
    127
    <type 'numpy.int16'> :
    -32768
    32767
    <type 'numpy.int32'> :
    -2147483648
    2147483647
    <type 'numpy.int64'> :
    -9223372036854775808
    9223372036854775807
    <type 'numpy.float16'> :
    -65504.0
    65504.0
    0.00097656
    <type 'numpy.float32'> :
    -3.40282e+38
    3.40282e+38
    1.19209e-07
    <type 'numpy.float64'> :
    -1.79769313486e+308
    1.79769313486e+308
    2.22044604925e-16
    <type 'numpy.float128'> :
    -inf
    inf
    1.08420217249e-19
    

#### 49. How to print all the values of an array? (★★☆)


```python
# Set printing options
# threshold, Total number of array elements which trigger summarization 
#            rather than full repr (default 1000).
np.set_printoptions(threshold=np.nan)
Z = np.zeros((16,16))
print(Z)
```

#### 50. How to find the closest value (to a given scalar) in a vector? (★★☆)


```python
a = np.random.randint(0,100,10)
x = 50
print a
temp = np.abs(a - x)
print a[temp.argmin()]
```

    [76 91 96 23 40 10 37 98  1 42]
    42
    

#### 51. Create a structured array representing a position (x,y) and a color (r,g,b) (★★☆)


```python
postion = np.dtype([('x', float, 1), ('y', float, 1)])
color = np.dtype([('r', float, 1), ('g', float, 1), ('b', float, 1)])
print postion, color
X = np.zeros(10, dtype=[('postion', postion), ('color', color)])
Y = np.zeros(10, dtype=[ ('position', [ ('x', float, 1),
                                  ('y', float, 1)]),
                   ('color',    [ ('r', float, 1),
                                  ('g', float, 1),
                                  ('b', float, 1)])])
print X.dtype
print Y.dtype
```

    [('x', '<f8'), ('y', '<f8')] [('r', '<f8'), ('g', '<f8'), ('b', '<f8')]
    [('postion', [('x', '<f8'), ('y', '<f8')]), ('color', [('r', '<f8'), ('g', '<f8'), ('b', '<f8')])]
    [('position', [('x', '<f8'), ('y', '<f8')]), ('color', [('r', '<f8'), ('g', '<f8'), ('b', '<f8')])]
    

#### 52. Consider a random vector with shape (100,2) representing coordinates, find point by point distances (★★☆)


```python
a = np.random.random((5,2))
print a
# View inputs as arrays with at least two dimensions.
# Parameters: arys1, arys2, ... : array_like
X,Y = np.atleast_2d(a[:,0], a[:,1])
print np.sqrt((X - X.T) ** 2 + (Y - Y.T) ** 2)

# Much faster with scipy
# Thanks Gavin Heverly-Coulson (#issue 1)
import scipy
import scipy.spatial
D = scipy.spatial.distance.cdist(a,a)
print(D)
```

    [[ 0.82233346  0.95837276]
     [ 0.99355572  0.34667307]
     [ 0.54259358  0.4680786 ]
     [ 0.96033521  0.58903852]
     [ 0.70558786  0.60051696]]
    [[ 0.          0.63521144  0.56448451  0.39427435  0.37641773]
     [ 0.63521144  0.          0.46701836  0.24463159  0.38387785]
     [ 0.56448451  0.46701836  0.          0.43490157  0.2100168 ]
     [ 0.39427435  0.24463159  0.43490157  0.          0.25500582]
     [ 0.37641773  0.38387785  0.2100168   0.25500582  0.        ]]
    [[ 0.          0.63521144  0.56448451  0.39427435  0.37641773]
     [ 0.63521144  0.          0.46701836  0.24463159  0.38387785]
     [ 0.56448451  0.46701836  0.          0.43490157  0.2100168 ]
     [ 0.39427435  0.24463159  0.43490157  0.          0.25500582]
     [ 0.37641773  0.38387785  0.2100168   0.25500582  0.        ]]
    

#### 53. How to convert a float (32 bits) array into an integer (32 bits) in place?


```python
a = np.arange(10, dtype=np.float32)
print a.dtype
a = a.astype(np.int32, copy=0)
print a.dtype
```

    float32
    int32
    

#### 54. How to read the following file? (★★☆)

```
1, 2, 3, 4, 5
6,  ,  , 7, 8
 ,  , 9,10,11
```


```python
# numpy.genfromtxt: Load data from a text file, with missing values handled as specified.
# refer to: https://docs.scipy.org/doc/numpy/reference/generated/numpy.genfromtxt.html
a = np.genfromtxt('input1.txt', delimiter=',')
print a
```

    [[  1.   2.   3.   4.   5.]
     [  6.  nan  nan   7.   8.]
     [ nan  nan   9.  10.  11.]]
    

#### 55. What is the equivalent of enumerate for numpy arrays? (★★☆)


```python
# enumerate in python can get index and value at the same time
Z = np.arange(9).reshape(3,3)
for index, value in np.ndenumerate(Z):
    print index, value 
# for index in np.ndindex(Z.shape):
#     print index, Z[index]
```

    (0, 0) 0
    (0, 1) 1
    (0, 2) 2
    (1, 0) 3
    (1, 1) 4
    (1, 2) 5
    (2, 0) 6
    (2, 1) 7
    (2, 2) 8
    

#### 56. Generate a generic 2D Gaussian-like array (★★☆)


```python
X,Y = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
# print X
# print Y
D = np.sqrt(X * X + Y * Y)
sigma, mu = 1.0, 0.0
G = np.exp(-((D - mu) ** 2 / (2.0 * sigma ** 2)))
print G
```

    [[ 0.36787944  0.44822088  0.51979489  0.57375342  0.60279818  0.60279818
       0.57375342  0.51979489  0.44822088  0.36787944]
     [ 0.44822088  0.54610814  0.63331324  0.69905581  0.73444367  0.73444367
       0.69905581  0.63331324  0.54610814  0.44822088]
     [ 0.51979489  0.63331324  0.73444367  0.81068432  0.85172308  0.85172308
       0.81068432  0.73444367  0.63331324  0.51979489]
     [ 0.57375342  0.69905581  0.81068432  0.89483932  0.9401382   0.9401382
       0.89483932  0.81068432  0.69905581  0.57375342]
     [ 0.60279818  0.73444367  0.85172308  0.9401382   0.98773022  0.98773022
       0.9401382   0.85172308  0.73444367  0.60279818]
     [ 0.60279818  0.73444367  0.85172308  0.9401382   0.98773022  0.98773022
       0.9401382   0.85172308  0.73444367  0.60279818]
     [ 0.57375342  0.69905581  0.81068432  0.89483932  0.9401382   0.9401382
       0.89483932  0.81068432  0.69905581  0.57375342]
     [ 0.51979489  0.63331324  0.73444367  0.81068432  0.85172308  0.85172308
       0.81068432  0.73444367  0.63331324  0.51979489]
     [ 0.44822088  0.54610814  0.63331324  0.69905581  0.73444367  0.73444367
       0.69905581  0.63331324  0.54610814  0.44822088]
     [ 0.36787944  0.44822088  0.51979489  0.57375342  0.60279818  0.60279818
       0.57375342  0.51979489  0.44822088  0.36787944]]
    

#### 57. How to randomly place p elements in a 2D array? (★★☆)


```python
# np.put(array_like_to_put, flat_index, array_like_put_num, mode=None)
## Replaces specified elements of an array with given values.
## The indexing works on the flattened target array. put is roughly equivalent to: a.flat[ind] = v

# numpy.random.choice(a, size=None, replace=True, p=None)
## Generates a random sample from a given 1-D array
## replace: Whether the sample is with or without replacement
a = np.zeros((10,10))
p = 3
np.put(a, np.random.choice(range(10*10), p, replace=False), 1)
print a
```

    [[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  1.  0.  0.  1.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  1.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]]
    

#### 58. Subtract the mean of each row of a matrix (★★☆)


```python
a = np.random.randint(0,100, 5*8).reshape(5, 8)
print a
mean = a.mean(axis=1, keepdims=True)
print a - mean
```

    [[43 23 17 20 89 17  0 43]
     [67 98  9 46 22  2 30 40]
     [ 5 41 64 41 94 49 42 60]
     [55 63 25 36 55 12 87 82]
     [41 67 52 65 36 46 10 64]]
    [[ 11.5    -8.5   -14.5   -11.5    57.5   -14.5   -31.5    11.5  ]
     [ 27.75   58.75  -30.25    6.75  -17.25  -37.25   -9.25    0.75 ]
     [-44.5    -8.5    14.5    -8.5    44.5    -0.5    -7.5    10.5  ]
     [  3.125  11.125 -26.875 -15.875   3.125 -39.875  35.125  30.125]
     [ -6.625  19.375   4.375  17.375 -11.625  -1.625 -37.625  16.375]]
    

#### 59. How to sort an array by the nth column? (★★☆)


```python
# np.argsort(a, axis=-1, kind='quicksort', order=None)
# Returns the indices that would sort an array
# order : str or list of str, optional
## When a is an array with fields defined
## this argument specifies which fields to compare first, second, etc. 
a = np.random.randint(0,100, (5, 8))
print a
a.sort(axis=0)
print a
a.sort(axis=1)
print a
```

    [[39 34 52 64 34 13 55 25]
     [87 36 18 49 10  2 80 64]
     [ 2 37 16  5 17 27 20 91]
     [38 89 19  0 30  4 82 60]
     [60 80 59 88  8 56 14 23]]
    [[ 2 34 16  0  8  2 14 23]
     [38 36 18  5 10  4 20 25]
     [39 37 19 49 17 13 55 60]
     [60 80 52 64 30 27 80 64]
     [87 89 59 88 34 56 82 91]]
    [[ 0  2  2  8 14 16 23 34]
     [ 4  5 10 18 20 25 36 38]
     [13 17 19 37 39 49 55 60]
     [27 30 52 60 64 64 80 80]
     [34 56 59 82 87 88 89 91]]
    

#### 60. How to tell if a given 2D array has null columns? (★★☆)


```python
a = np.random.randint(0, 3, (3, 10))
print a
print a.any(axis=0)
print ((~a.any(axis=0)).any())
```

    [[0 1 2 1 1 1 1 1 0 0]
     [0 1 1 0 1 0 0 0 0 0]
     [2 1 1 2 0 1 2 0 0 0]]
    [ True  True  True  True  True  True  True  True False False]
    True
    

#### 61. Find the nearest value from a given value in an array (★★☆)


```python
a = np.random.randint(0, 100, 10)
print a
x = 6
res = a[np.argmin(np.abs(a-x))]
print res
```

    [46 11 19 46  7 18 14 14 97 87]
    7
    

#### 62. Considering two arrays with shape (1,3) and (3,1), how to compute their sum using an iterator? (★★☆)


```python
a = np.random.randint(1, 10, (1,3))
b = np.random.randint(1, 10, (3,1))
print a
print b
it = np.nditer([a,b,None])
print it
for x,y,z in it: 
    print x, y, z
    z[...] = x + y
print(it.operands[2])
```

    [[2 4 2]]
    [[3]
     [9]
     [8]]
    <numpy.nditer object at 0x7f1902a2cf30>
    2 3 15
    4 3 16
    2 3 14
    2 9 16
    4 9 17
    2 9 15
    2 8 13
    4 8 14
    2 8 12
    [[ 5  7  5]
     [11 13 11]
     [10 12 10]]
    

#### 63. Create an array class that has a name attribute (★★☆)


```python
class NameArray(np.ndarray):
    def __new__(cls, array, name="no name"):
        obj = np.asarray(array).view(cls)
        obj.name = name
        return obj
    def __array_finalize(self, obj):
        if obj is None:
            return
        self.info = getattr(obj, 'name', "no name")
        
Z = NameArray(np.arange(10), "range_10")
print Z.name
```

    range_10
    

#### 64. Consider a given vector, how to add 1 to each element indexed by a second vector (be careful with repeated indices)? (★★★)


```python
a = np.zeros(10)
idx = [2, 5, 9, 2]
a[idx] += 1
print a
np.add.at(a, idx, 1)
print(a)
```

    [ 0.  0.  1.  0.  0.  1.  0.  0.  0.  1.]
    [ 0.  0.  3.  0.  0.  2.  0.  0.  0.  2.]
    

#### 65. How to accumulate elements of a vector (X) to an array (F) based on an index list (I)? (★★★)


```python
# numpy.bincount(x, weights=None, minlength=0)
# Count number of occurrences of each value in array of non-negative ints
## the return array res[0...x.max()]
# default: the res[x[idx]] += 1, the weights make res[x[idx]] += weights[idx]
I = [5,2,8,4,6,1,5,2,9,5]
X = np.arange(10)
F = np.bincount(I, X)
print I
print X
print F
```

    [5, 2, 8, 4, 6, 1, 5, 2, 9, 5]
    [0 1 2 3 4 5 6 7 8 9]
    [  0.   5.   8.   0.   3.  15.   4.   0.   2.   8.]
    

#### 66. Considering a (w,h,3) image of (dtype=ubyte), compute the number of unique colors (★★★)


```python
I = np.random.randint(0, 2, (5,5,3)).astype(np.ubyte)
F = I[...,0]*256*256 + I[...,1]*256 +I[...,2]
print F
n = len(np.unique(F))
print(n)
```

    [[  0 256   0 257 257]
     [256 256   1 257   0]
     [257   1   1   0   0]
     [257   0 257 257 256]
     [  1   1   1 257 256]]
    4
    

#### 67. Considering a four dimensions array, how to get sum over the last two axis at once? (★★★)


```python
a = np.random.randint(0, 100, (5,5,2,3))
# print a
print a.sum(axis=(-2,-1))
```

    [[221 395 245 117 333]
     [413 249 193 251 344]
     [301 254 403 361 213]
     [368 305 317 413 304]
     [284 291 326 253 339]]
    

#### 68. Considering a one-dimensional vector D, how to compute means of subsets of D using a vector S of same size describing subset  indices? (★★★)


```python
# the S contains repeated indexes, each element occurs with different weights in D
# we can take [0,10] as class and elements of D as samples. The number of samples is 100
# this request is to compute the mean of each class
D = np.random.uniform(0,1,100)
print "D=", D
S = np.random.randint(0,10,100)
print "S=",  S
D_sums = np.bincount(S, weights=D)
print "D_sums=", D_sums
D_counts = np.bincount(S)
print "D_counts=", D_counts
D_means = D_sums / D_counts
print(D_means)

```

    D= [ 0.88666172  0.34304711  0.4258038   0.69523787  0.23173362  0.40101973
      0.44245461  0.88937984  0.50294445  0.74747347  0.75210929  0.27437857
      0.62513244  0.87519701  0.64726227  0.2813883   0.63354429  0.66428191
      0.48062449  0.23273264  0.32475683  0.29123075  0.99208337  0.86388675
      0.08835137  0.8438834   0.04188953  0.32683276  0.20119604  0.60866828
      0.56151865  0.40865062  0.33106821  0.00522547  0.93576187  0.30003982
      0.20595365  0.27664135  0.61553513  0.4883546   0.20020938  0.66906914
      0.4686936   0.47557879  0.21283174  0.1269313   0.37053397  0.92840811
      0.18883938  0.83859443  0.12193762  0.82507892  0.17220598  0.27138704
      0.24053606  0.98516997  0.95933826  0.14819655  0.53619272  0.78588692
      0.01984188  0.8101043   0.92368572  0.3717517   0.48395408  0.2213148
      0.29218366  0.36455274  0.37733882  0.6835993   0.40728883  0.86030458
      0.03351963  0.34826479  0.48594057  0.7990572   0.74444969  0.89178264
      0.8405332   0.88536493  0.32397707  0.0990161   0.00444753  0.66418555
      0.92141371  0.91833689  0.94219759  0.45792362  0.99090376  0.53107708
      0.92872616  0.3325592   0.75926188  0.59764164  0.09936018  0.23302551
      0.8096037   0.36360976  0.99208948  0.2919603 ]
    S= [3 9 2 7 8 0 3 3 4 1 5 7 2 8 0 4 2 1 9 4 8 5 7 2 3 0 9 1 7 4 3 9 8 7 1 6 0
     2 4 1 8 1 8 0 5 8 9 1 4 6 2 4 5 5 0 9 0 2 0 2 4 9 9 3 2 2 9 7 0 9 4 0 1 0
     6 8 8 9 8 5 6 3 4 4 5 1 3 8 0 1 1 7 8 1 9 9 9 6 8 1]
    D_sums= [ 6.88657682  8.06144354  4.58629858  4.28133159  4.35095089  3.50654342
      2.31216166  2.86523326  7.151905    7.37326071]
    D_counts= [12 13 10  8 11  7  5  7 13 14]
    [ 0.5738814   0.62011104  0.45862986  0.53516645  0.39554099  0.50093477
      0.46243233  0.40931904  0.55014654  0.52666148]
    

#### 69. How to get the diagonal of a dot product? (★★★)


```python
A = np.random.uniform(0,1,(5,5))
B = np.random.uniform(0,1,(5,5))
print np.diag(np.dot(A, B))
# Fast version
print np.sum(A * B.T, axis=1)
# Faster version
print np.einsum("ij,ji->i", A, B)
```

    [ 1.82413289  1.0860061   1.27552764  0.62973688  1.64674707]
    [ 1.82413289  1.0860061   1.27552764  0.62973688  1.64674707]
    [ 1.82413289  1.0860061   1.27552764  0.62973688  1.64674707]
    

#### 70. Consider the vector \[1, 2, 3, 4, 5\], how to build a new vector with 3 consecutive zeros interleaved between each value? (★★★)


```python
Z = np.array([1,2,3,4,5])
nz = 3
Z0 = np.zeros(len(Z) + (len(Z)-1)*(nz))
Z0[::nz+1] = Z
print(Z0)
```

    [ 1.  0.  0.  0.  2.  0.  0.  0.  3.  0.  0.  0.  4.  0.  0.  0.  5.]
    

#### 71. Consider an array of dimension (5,5,3), how to mulitply it by an array with dimensions (5,5)? (★★★)


```python
A = np.ones((5,5,3))
B = 2*np.ones((5,5))
# add one dim
print B[:,:,None].shape #(5, 5, 1)
print (A * B[:,:,None]).shape #(5, 5, 3)
```

    (5, 5, 1)
    (5, 5, 3)
    

#### 72. How to swap two rows of an array? (★★★)


```python
A = np.arange(25).reshape(5,5)
print A
# A[(0,1)] = A[(1,0)] () refer to the elements, this doesn't swap but copy
A[:,[0,1]] = A[:,[1,0]] # [] refer to the row
print A
```

    [[ 0  1  2  3  4]
     [ 5  6  7  8  9]
     [10 11 12 13 14]
     [15 16 17 18 19]
     [20 21 22 23 24]]
    [[ 1  0  2  3  4]
     [ 6  5  7  8  9]
     [11 10 12 13 14]
     [16 15 17 18 19]
     [21 20 22 23 24]]
    

#### 73. Consider a set of 10 triplets describing 10 triangles (with shared vertices), find the set of unique line segments composing all the  triangles (★★★)


```python
faces = np.random.randint(0, 100, (10, 3))
print faces
print faces.repeat(2, axis=1)
# roll(array, shift, axis)
# shift by ring, right > 0 and left < 0
F = np.roll(faces.repeat(2, axis=1), -1, axis=1)
print F
F = F.reshape(len(F)*3, 2)
print F #edge
G = F.view(dtype=[('p0', F.dtype), ('p1', F.dtype)])
G = np.unique(G)
print G
```

    [[25 95 90]
     [43 97 27]
     [98 21 81]
     [41 14 85]
     [79  1  0]
     [62 13 44]
     [29 61 33]
     [66 96 66]
     [44 61 13]
     [52  7 96]]
    [[25 25 95 95 90 90]
     [43 43 97 97 27 27]
     [98 98 21 21 81 81]
     [41 41 14 14 85 85]
     [79 79  1  1  0  0]
     [62 62 13 13 44 44]
     [29 29 61 61 33 33]
     [66 66 96 96 66 66]
     [44 44 61 61 13 13]
     [52 52  7  7 96 96]]
    [[25 95 95 90 90 25]
     [43 97 97 27 27 43]
     [98 21 21 81 81 98]
     [41 14 14 85 85 41]
     [79  1  1  0  0 79]
     [62 13 13 44 44 62]
     [29 61 61 33 33 29]
     [66 96 96 66 66 66]
     [44 61 61 13 13 44]
     [52  7  7 96 96 52]]
    [[25 95]
     [95 90]
     [90 25]
     [43 97]
     [97 27]
     [27 43]
     [98 21]
     [21 81]
     [81 98]
     [41 14]
     [14 85]
     [85 41]
     [79  1]
     [ 1  0]
     [ 0 79]
     [62 13]
     [13 44]
     [44 62]
     [29 61]
     [61 33]
     [33 29]
     [66 96]
     [96 66]
     [66 66]
     [44 61]
     [61 13]
     [13 44]
     [52  7]
     [ 7 96]
     [96 52]]
    [( 0, 79) ( 1,  0) ( 7, 96) (13, 44) (14, 85) (21, 81) (25, 95) (27, 43)
     (29, 61) (33, 29) (41, 14) (43, 97) (44, 61) (44, 62) (52,  7) (61, 13)
     (61, 33) (62, 13) (66, 66) (66, 96) (79,  1) (81, 98) (85, 41) (90, 25)
     (95, 90) (96, 52) (96, 66) (97, 27) (98, 21)]
    

#### 74. Given an array C that is a bincount, how to produce an array A such that np.bincount(A) == C? (★★★)


```python
C = np.random.randint(0, 10, 10)
print C
A = np.repeat(np.arange(len(C)), C)
print A
```

    [7 8 1 4 1 0 3 2 5 9]
    [0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 2 3 3 3 3 4 6 6 6 7 7 8 8 8 8 8 9 9 9 9 9 9
     9 9 9]
    

#### 75. How to compute averages using a sliding window over an array? (★★★)


```python
from __future__ import division
def mv_avg(a, n = 3):
    ret = np.cumsum(a, dtype=float)
    print ret
    ret[n:] = ret[n:] - ret[:-n]
    print ret
    return ret[n-1:]/n
Z = np.arange(20)
print mv_avg(Z)
```

    [   0.    1.    3.    6.   10.   15.   21.   28.   36.   45.   55.   66.
       78.   91.  105.  120.  136.  153.  171.  190.]
    [  0.   1.   3.   6.   9.  12.  15.  18.  21.  24.  27.  30.  33.  36.  39.
      42.  45.  48.  51.  54.]
    [  1.   2.   3.   4.   5.   6.   7.   8.   9.  10.  11.  12.  13.  14.  15.
      16.  17.  18.]
    

#### 76. Consider a one-dimensional array Z, build a two-dimensional array whose first row is (Z\[0\],Z\[1\],Z\[2\]) and each subsequent row is  shifted by 1 (last row should be (Z\[-3\],Z\[-2\],Z\[-1\]) (★★★)


```python
Z = np.arange(10)
step = 3
row = len(Z) - step + 1
F = np.zeros((row, step))
for i in range(row):
    F[i,:] = Z[i:i+3]
print F

from numpy.lib import stride_tricks

def rolling(a, window):
    shape = (a.size - window + 1, window)
    strides = (a.itemsize, a.itemsize)
    return stride_tricks.as_strided(a, shape=shape, strides=strides)
Z = rolling(np.arange(10), 3)
print(Z)
```

    [[ 0.  1.  2.]
     [ 1.  2.  3.]
     [ 2.  3.  4.]
     [ 3.  4.  5.]
     [ 4.  5.  6.]
     [ 5.  6.  7.]
     [ 6.  7.  8.]
     [ 7.  8.  9.]]
    [[0 1 2]
     [1 2 3]
     [2 3 4]
     [3 4 5]
     [4 5 6]
     [5 6 7]
     [6 7 8]
     [7 8 9]]
    

#### 77. How to negate a boolean, or to change the sign of a float inplace? (★★★)


```python
flag = np.random.randint(0, 2, 100)
print np.logical_not(flag, out = flag)
f = np.random.uniform(0, 1, 100)
print np.negative(f, out = f)
```

    [1 1 0 0 0 0 0 0 1 0 0 0 1 0 1 0 0 0 0 1 1 0 0 1 0 1 0 1 1 0 1 1 1 1 1 0 0
     0 1 1 0 0 0 1 0 0 1 1 0 0 0 1 0 0 1 0 1 1 1 0 0 0 0 0 1 0 0 1 0 0 1 0 1 1
     0 1 1 0 1 0 0 1 0 1 1 0 1 0 0 1 1 0 1 0 1 0 1 0 1 1]
    [-0.35813854 -0.91302982 -0.00349167 -0.6274243  -0.92602243 -0.28615085
     -0.96706885 -0.52893999 -0.24229335 -0.62091889 -0.65778484 -0.44439223
     -0.34615486 -0.22510695 -0.62752018 -0.08835982 -0.3095795  -0.27354945
     -0.00124274 -0.88445232 -0.74815699 -0.8378763  -0.73798509 -0.05773402
     -0.89580065 -0.78946917 -0.16040129 -0.22347283 -0.0100545  -0.949977
     -0.39215579 -0.73633173 -0.49908662 -0.38365928 -0.22306469 -0.12204318
     -0.25907161 -0.20504262 -0.27526769 -0.97262493 -0.88657729 -0.66453926
     -0.29561089 -0.10290164 -0.20802169 -0.65024979 -0.84484113 -0.32292215
     -0.29050417 -0.66698978 -0.01271162 -0.84332089 -0.34663672 -0.76942961
     -0.66589724 -0.73861717 -0.88886028 -0.21493973 -0.55541859 -0.76721265
     -0.66892131 -0.19487779 -0.50333448 -0.31232295 -0.05762797 -0.54784713
     -0.79788835 -0.70244226 -0.75053204 -0.8144552  -0.89205178 -0.54539158
     -0.65513685 -0.66685326 -0.70071493 -0.63841878 -0.02819422 -0.81148089
     -0.52235952 -0.85938555 -0.9716362  -0.68945428 -0.41962865 -0.43870541
     -0.92767819 -0.42659806 -0.06534491 -0.28208632 -0.11759915 -0.74377482
     -0.93807123 -0.37054151 -0.86060487 -0.01728764 -0.85660868 -0.93305487
     -0.03367833 -0.70585417 -0.591582   -0.6276782 ]
    

#### 78. Consider 2 sets of points P0,P1 describing lines (2d) and a point p, how to compute distance from p to each line i  (P0\[i\],P1\[i\])? (★★★)


```python
p = np.random.uniform(0, 1, (1, 2))
P0 = np.random.uniform(0, 1, (10,2))
P1 = np.random.uniform(0, 1, (10,2))

# d = |Ax-by+C| / sqrt(A**2+B**2)
# A = y1 - y2
# B = x1 - x2
# C = x1y2 - x2y1
AB = P0 - P1
C = P0[:,0] * P1[:,1] - P0[:,1] * P1[:,0]
print AB.shape
down = np.sqrt(np.sum(AB ** 2, axis = 1))
print down.shape
up = np.abs(AB[:,1] * p[0][0] - AB[:,0] * p[0][1] + C)
print up.shape
dist = up / down
print dist

# def distance(P0, P1, p):
#     T = P1 - P0
#     L = (T**2).sum(axis=1)
#     U = -((P0[:,0]-p[...,0])*T[:,0] + (P0[:,1]-p[...,1])*T[:,1]) / L
#     U = U.reshape(len(U),1)
#     D = P0 + U*T - p
#     return np.sqrt((D**2).sum(axis=1))
# print(distance(P0, P1, p))
```

    (10, 2)
    (10,)
    (10,)
    [ 0.02238501  0.30350078  0.30112154  0.15846085  0.02638434  0.06482239
      0.2386688   0.03744339  0.66166379  0.20082729]
    [ 0.02238501  0.30350078  0.30112154  0.15846085  0.02638434  0.06482239
      0.2386688   0.03744339  0.66166379  0.20082729]
    

#### 79. Consider 2 sets of points P0,P1 describing lines (2d) and a set of points P, how to compute distance from each point j (P\[j\]) to each line i (P0\[i\],P1\[i\])? (★★★)


```python
p = np.random.uniform(0, 1, (5, 2))
P0 = np.random.uniform(0, 1, (10,2))
P1 = np.random.uniform(0, 1, (10,2))

AB = P0 - P1
C = P0[:,0] * P1[:,1] - P0[:,1] * P1[:,0]
print AB.shape
down = np.sqrt(np.sum(AB ** 2, axis = 1))
print down.shape
B, A = np.atleast_2d(AB[:,0], AB[:,1])
X, Y = np.atleast_2d(p[:, 0], p[:, 1])
up = np.abs(np.dot(X.T, A) - np.dot(Y.T,B) + C)
print up.shape
dist = up / down
print dist

print(np.array([distance(P0,P1,p_i) for p_i in p]))
```

    (10, 2)
    (10,)
    (5, 10)
    [[ 0.49700483  0.34504314  0.87402843  0.74897864  0.46003239  0.53463393
       0.09711807  0.07997124  0.19655399  0.15780491]
     [ 0.2558936   0.16520677  0.18595152  0.03978621  0.04886974  0.05890842
       0.29147476  0.09623676  0.22094737  0.56026379]
     [ 0.49725281  0.2675272   0.16707136  0.03642137  0.67689389  0.59519848
       0.26659052  0.84971089  0.31127498  0.63727291]
     [ 0.47470817  0.39685124  0.89814304  0.79144241  0.45994403  0.5312773
       0.14773573  0.11579893  0.24764693  0.19720487]
     [ 0.28907706  0.34229185  0.66823467  0.61465286  0.22865504  0.3038786
       0.14432765  0.24561263  0.23274157  0.00591675]]
    [[ 0.49700483  0.34504314  0.87402843  0.74897864  0.46003239  0.53463393
       0.09711807  0.07997124  0.19655399  0.15780491]
     [ 0.2558936   0.16520677  0.18595152  0.03978621  0.04886974  0.05890842
       0.29147476  0.09623676  0.22094737  0.56026379]
     [ 0.49725281  0.2675272   0.16707136  0.03642137  0.67689389  0.59519848
       0.26659052  0.84971089  0.31127498  0.63727291]
     [ 0.47470817  0.39685124  0.89814304  0.79144241  0.45994403  0.5312773
       0.14773573  0.11579893  0.24764693  0.19720487]
     [ 0.28907706  0.34229185  0.66823467  0.61465286  0.22865504  0.3038786
       0.14432765  0.24561263  0.23274157  0.00591675]]
    

#### 80. Consider an arbitrary array, write a function that extract a subpart with a fixed shape and centered on a given element (pad with a `fill` value when necessary) (★★★)


```python
Z = np.random.randint(0,10,(10,10))
shape = (5,5)
fill  = 0
position = (1,1)

R = np.ones(shape, dtype=Z.dtype)*fill
P  = np.array(list(position)).astype(int)
Rs = np.array(list(R.shape)).astype(int)
Zs = np.array(list(Z.shape)).astype(int)

R_start = np.zeros((len(shape),)).astype(int)
R_stop  = np.array(list(shape)).astype(int)
Z_start = (P-Rs//2)
Z_stop  = (P+Rs//2)+Rs%2

R_start = (R_start - np.minimum(Z_start,0)).tolist()
Z_start = (np.maximum(Z_start,0)).tolist()
R_stop = np.maximum(R_start, (R_stop - np.maximum(Z_stop-Zs,0))).tolist()
Z_stop = (np.minimum(Z_stop,Zs)).tolist()

r = [slice(start,stop) for start,stop in zip(R_start,R_stop)]
z = [slice(start,stop) for start,stop in zip(Z_start,Z_stop)]
R[r] = Z[z]
print(Z)
print(R)
```

#### 81. Consider an array Z = \[1,2,3,4,5,6,7,8,9,10,11,12,13,14\], how to generate an array R = \[\[1,2,3,4\], \[2,3,4,5\], \[3,4,5,6\], ..., \[11,12,13,14\]\]? (★★★)


```python
from numpy.lib import stride_tricks
# numpy.lib.stride_tricks.as_strided(x, shape=None, strides=None, subok=False, writeable=True)
## strides: The strides of the new array, strides = (a.itemsize, a.itemsize)
Z = np.arange(1,15,dtype=np.uint32)
R = stride_tricks.as_strided(Z,(11,4),(4,4))
print(R)
```

    [[ 1  2  3  4]
     [ 2  3  4  5]
     [ 3  4  5  6]
     [ 4  5  6  7]
     [ 5  6  7  8]
     [ 6  7  8  9]
     [ 7  8  9 10]
     [ 8  9 10 11]
     [ 9 10 11 12]
     [10 11 12 13]
     [11 12 13 14]]
    

#### 82. Compute a matrix rank (★★★)


```python
A = np.random.randint(0, 100, (7, 8))
print A
U, S, V = np.linalg.svd(A)
rank = np.sum(S > 1e-10)
print(rank)
print np.linalg.matrix_rank(A)
```

    [[95  1 96 10  7 29 42 95]
     [72 51 13 65 26 83 96 10]
     [12 78 99 19 44 93 20 13]
     [81 16 95 45 66 51  2 17]
     [66 41 53 73 81 98 21 87]
     [53 25  5 75  1 57 57 75]
     [64  5 70 20 39 16 75 68]]
    7
    7
    

#### 83. How to find the most frequent value in an array?


```python
Z = np.random.randint(0, 10, 100)
print Z
C = np.bincount(Z)
print C
print np.argmax(C)
```

    [1 6 7 3 5 4 1 9 5 4 4 8 5 8 9 1 7 3 3 4 1 5 4 5 5 5 5 2 4 5 9 9 4 2 8 9 1
     6 9 8 5 7 4 6 8 5 0 9 9 2 3 9 6 1 4 0 9 1 6 1 1 8 7 0 7 5 0 2 5 6 7 7 0 8
     8 1 5 0 7 4 4 2 7 0 0 8 4 5 9 5 7 4 5 5 0 0 7 8 8 6]
    [10 10  5  4 13 18  7 11 11 11]
    5
    

#### 84. Extract all the contiguous 3x3 blocks from a random 10x10 matrix (★★★)


```python
Z = np.random.randint(0, 10, (10, 10), dtype=np.int32)
print Z.strides # (40, 4) offset = sum(np.array(i) * a.strides)
print Z.strides + Z.strides #(40, 4, 40, 4)
print stride_tricks.as_strided(Z, shape=(8, 8, 3, 3), strides=Z.strides + Z.strides)
```

    (40, 4)
    (40, 4, 40, 4)
    [[[[9 7 0]
       [2 8 2]
       [7 7 0]]
    
      [[7 0 3]
       [8 2 5]
       [7 0 6]]
    
      [[0 3 9]
       [2 5 9]
       [0 6 9]]
    
      [[3 9 5]
       [5 9 9]
       [6 9 6]]
    
      [[9 5 6]
       [9 9 9]
       [9 6 0]]
    
      [[5 6 4]
       [9 9 4]
       [6 0 6]]
    
      [[6 4 2]
       [9 4 8]
       [0 6 2]]
    
      [[4 2 1]
       [4 8 5]
       [6 2 3]]]
    
    
     [[[2 8 2]
       [7 7 0]
       [5 9 8]]
    
      [[8 2 5]
       [7 0 6]
       [9 8 4]]
    
      [[2 5 9]
       [0 6 9]
       [8 4 5]]
    
      [[5 9 9]
       [6 9 6]
       [4 5 0]]
    
      [[9 9 9]
       [9 6 0]
       [5 0 2]]
    
      [[9 9 4]
       [6 0 6]
       [0 2 9]]
    
      [[9 4 8]
       [0 6 2]
       [2 9 3]]
    
      [[4 8 5]
       [6 2 3]
       [9 3 9]]]
    
    
     [[[7 7 0]
       [5 9 8]
       [7 5 8]]
    
      [[7 0 6]
       [9 8 4]
       [5 8 0]]
    
      [[0 6 9]
       [8 4 5]
       [8 0 6]]
    
      [[6 9 6]
       [4 5 0]
       [0 6 9]]
    
      [[9 6 0]
       [5 0 2]
       [6 9 7]]
    
      [[6 0 6]
       [0 2 9]
       [9 7 5]]
    
      [[0 6 2]
       [2 9 3]
       [7 5 6]]
    
      [[6 2 3]
       [9 3 9]
       [5 6 5]]]
    
    
     [[[5 9 8]
       [7 5 8]
       [0 9 8]]
    
      [[9 8 4]
       [5 8 0]
       [9 8 1]]
    
      [[8 4 5]
       [8 0 6]
       [8 1 7]]
    
      [[4 5 0]
       [0 6 9]
       [1 7 0]]
    
      [[5 0 2]
       [6 9 7]
       [7 0 9]]
    
      [[0 2 9]
       [9 7 5]
       [0 9 6]]
    
      [[2 9 3]
       [7 5 6]
       [9 6 8]]
    
      [[9 3 9]
       [5 6 5]
       [6 8 3]]]
    
    
     [[[7 5 8]
       [0 9 8]
       [4 7 7]]
    
      [[5 8 0]
       [9 8 1]
       [7 7 9]]
    
      [[8 0 6]
       [8 1 7]
       [7 9 1]]
    
      [[0 6 9]
       [1 7 0]
       [9 1 8]]
    
      [[6 9 7]
       [7 0 9]
       [1 8 2]]
    
      [[9 7 5]
       [0 9 6]
       [8 2 9]]
    
      [[7 5 6]
       [9 6 8]
       [2 9 0]]
    
      [[5 6 5]
       [6 8 3]
       [9 0 4]]]
    
    
     [[[0 9 8]
       [4 7 7]
       [8 7 1]]
    
      [[9 8 1]
       [7 7 9]
       [7 1 0]]
    
      [[8 1 7]
       [7 9 1]
       [1 0 2]]
    
      [[1 7 0]
       [9 1 8]
       [0 2 3]]
    
      [[7 0 9]
       [1 8 2]
       [2 3 8]]
    
      [[0 9 6]
       [8 2 9]
       [3 8 1]]
    
      [[9 6 8]
       [2 9 0]
       [8 1 9]]
    
      [[6 8 3]
       [9 0 4]
       [1 9 3]]]
    
    
     [[[4 7 7]
       [8 7 1]
       [4 5 5]]
    
      [[7 7 9]
       [7 1 0]
       [5 5 9]]
    
      [[7 9 1]
       [1 0 2]
       [5 9 2]]
    
      [[9 1 8]
       [0 2 3]
       [9 2 3]]
    
      [[1 8 2]
       [2 3 8]
       [2 3 2]]
    
      [[8 2 9]
       [3 8 1]
       [3 2 1]]
    
      [[2 9 0]
       [8 1 9]
       [2 1 7]]
    
      [[9 0 4]
       [1 9 3]
       [1 7 0]]]
    
    
     [[[8 7 1]
       [4 5 5]
       [3 0 9]]
    
      [[7 1 0]
       [5 5 9]
       [0 9 3]]
    
      [[1 0 2]
       [5 9 2]
       [9 3 1]]
    
      [[0 2 3]
       [9 2 3]
       [3 1 4]]
    
      [[2 3 8]
       [2 3 2]
       [1 4 5]]
    
      [[3 8 1]
       [3 2 1]
       [4 5 2]]
    
      [[8 1 9]
       [2 1 7]
       [5 2 1]]
    
      [[1 9 3]
       [1 7 0]
       [2 1 0]]]]
    

#### 85. Create a 2D array subclass such that Z\[i,j\] == Z\[j,i\] (★★★)


```python
# Note: only works for 2d array and value setting using indices

class Symetric(np.ndarray):
    def __setitem__(self, index, value):
        i,j = index
        super(Symetric, self).__setitem__((i,j), value)
        super(Symetric, self).__setitem__((j,i), value)

def symetric(Z):
    return np.asarray(Z + Z.T - np.diag(Z.diagonal())).view(Symetric)

S = symetric(np.random.randint(0,10,(5,5)))
S[2,3] = 42
print(S)
```

    [[ 6  7  9 11  8]
     [ 7  6 14 15  1]
     [ 9 14  5 42 10]
     [11 15 42  1  9]
     [ 8  1 10  9  2]]
    

#### 86. Consider a set of p matrices wich shape (n,n) and a set of p vectors with shape (n,1). How to compute the sum of of the p matrix products at once? (result has shape (n,1)) (★★★)


```python
# numpy.tensordot(a, b, axes=2)
## Compute tensor dot product along specified axes for arrays >= 1-D.
## # The third argument can be a single non-negative integer_like scalar, N;
## # if it is such, then the last N dimensions of a and the first N dimensions of b are summed over.
n = 5
p = 3
A = np.random.randint(0, 10, (p, n, n))
B = np.random.randint(0, 10, (p, n, 1))
print np.tensordot(A, B, axes=[[0, 2], [0, 1]])

# It works, because:
# M is (p,n,n)
# V is (p,n,1)
# Thus, summing over the paired axes 0 and 0 (of M and V independently),
# and 2 and 1, to remain with a (n,1) vector.
```

    [[351]
     [343]
     [308]
     [267]
     [353]]
    

#### 87. Consider a 16x16 array, how to get the block-sum (block size is 4x4)? (★★★)


```python
Z = np.random.randint(0, 10, (16, 16), dtype=np.int32)
step = 4
intsize = 4
block_size = 16 / step
block =  stride_tricks.as_strided(Z, shape=(block_size, block_size, 4, 4), strides=(step*16*intsize, step*intsize, 16*intsize, intsize))
s = np.sum(block, axis = (2, 3))
print s.shape
print s

# ufunc.reduceat(a, indices, axis=0, dtype=None, out=None)
## refer to https://docs.scipy.org/doc/numpy/reference/generated/numpy.ufunc.reduceat.html
k = 4
# [0, k, 2k, 3k] means row0+row1+row2+row3,row4+row5+row6+row7
S = np.add.reduceat(np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                                       np.arange(0, Z.shape[1], k), axis=1)
print(S)
```

    (4, 4)
    [[ 66  82  75  65]
     [100  33  69  60]
     [ 57  72  66  87]
     [ 51  41  77  67]]
    [[ 66  82  75  65]
     [100  33  69  60]
     [ 57  72  66  87]
     [ 51  41  77  67]]
    

#### 88. How to implement the Game of Life using numpy arrays? (★★★)


```python
# the game of life can refer to http://www.cnblogs.com/grandyang/p/4854466.html
def iterate(Z):
    # Count neighbours
    N = (Z[0:-2,0:-2] + Z[0:-2,1:-1] + Z[0:-2,2:] +
         Z[1:-1,0:-2]                + Z[1:-1,2:] +
         Z[2:  ,0:-2] + Z[2:  ,1:-1] + Z[2:  ,2:])

    # Apply rules
    birth = (N==3) & (Z[1:-1,1:-1]==0) # dead
    survive = ((N==2) | (N==3)) & (Z[1:-1,1:-1]==1) # live
    Z[...] = 0
    Z[1:-1,1:-1][birth | survive] = 1
    return Z

Z = np.random.randint(0,2,(50,50))
for i in range(100): 
    Z = iterate(Z)
print(Z)
```

#### 89. How to get the n largest values of an array (★★★)


```python
Z = np.random.random(10)
print Z
ZS = np.sort(Z)
max_num = 5
print ZS[-5:]

n=5

# Slow
print (Z[np.argsort(Z)[-n:]])

# Fast
## numpy.argpartition(a, kth, axis=-1, kind='introselect', order=None)
## the element before kth is simaller than kth and those after kth is bigger than it 
print (Z[np.argpartition(-Z,n)[:n]])
```

    [ 0.24056592  0.75311397  0.93405731  0.6955102   0.14809363  0.6797921
      0.17999748  0.70125779  0.5398704   0.33891926]
    [ 0.6797921   0.6955102   0.70125779  0.75311397  0.93405731]
    [ 0.6797921   0.6955102   0.70125779  0.75311397  0.93405731]
    [ 0.75311397  0.70125779  0.93405731  0.6955102   0.6797921 ]
    

#### 90. Given an arbitrary number of vectors, build the cartesian product (every combinations of every item) (★★★)


```python
# numpy.indices(dimensions, dtype=<type 'int'>)
## Return an array representing the indices of a grid.
a = [1, 2, 3]
b = [4, 5]
c = [6, 7, 8, 9]

def cart(arrays):
    arrays = [np.asarray(a) for a in arrays]
    shape = (len(x) for x in arrays)
    
    ix = np.indices(shape, dtype=int)
    ix = ix.reshape(len(arrays), -1)  #(3, 3*2*4)
    
    res = np.zeros(ix.shape)
    for i in range(len(arrays)):
        res[i,:] = arrays[i][ix[i, :]]
        
    return res.T
    
print (cart((a, b, c)))    

def cartesian(arrays):
    arrays = [np.asarray(a) for a in arrays]
    shape = (len(x) for x in arrays)

    ix = np.indices(shape, dtype=int)
    ix = ix.reshape(len(arrays), -1).T

    for n, arr in enumerate(arrays):
        ix[:, n] = arrays[n][ix[:, n]]

    return ix

print (cartesian((a, b, c)))
```

    [[ 1.  4.  6.]
     [ 1.  4.  7.]
     [ 1.  4.  8.]
     [ 1.  4.  9.]
     [ 1.  5.  6.]
     [ 1.  5.  7.]
     [ 1.  5.  8.]
     [ 1.  5.  9.]
     [ 2.  4.  6.]
     [ 2.  4.  7.]
     [ 2.  4.  8.]
     [ 2.  4.  9.]
     [ 2.  5.  6.]
     [ 2.  5.  7.]
     [ 2.  5.  8.]
     [ 2.  5.  9.]
     [ 3.  4.  6.]
     [ 3.  4.  7.]
     [ 3.  4.  8.]
     [ 3.  4.  9.]
     [ 3.  5.  6.]
     [ 3.  5.  7.]
     [ 3.  5.  8.]
     [ 3.  5.  9.]]
    [[1 4 6]
     [1 4 7]
     [1 4 8]
     [1 4 9]
     [1 5 6]
     [1 5 7]
     [1 5 8]
     [1 5 9]
     [2 4 6]
     [2 4 7]
     [2 4 8]
     [2 4 9]
     [2 5 6]
     [2 5 7]
     [2 5 8]
     [2 5 9]
     [3 4 6]
     [3 4 7]
     [3 4 8]
     [3 4 9]
     [3 5 6]
     [3 5 7]
     [3 5 8]
     [3 5 9]]
    

#### 91. How to create a record array from a regular array? (★★★)


```python
# core.records.fromarrays: create a record array from a (flat) list of arrays
Z = np.array([("Hello", 2.5, 3),
              ("World", 3.6, 2)])
R = np.core.records.fromarrays(Z.T, 
                               names='col1, col2, col3',
                               formats = 'S8, f8, i8')
print(R)
print R.col1
print R[0]
```

    [('Hello', 2.5, 3L) ('World', 3.6, 2L)]
    ['Hello' 'World']
    ('Hello', 2.5, 3L)
    

#### 92. Consider a large vector Z, compute Z to the power of 3 using 3 different methods (★★★)


```python
Z = np.random.random(100000)
%timeit Z1 = Z * Z * Z
%timeit Z2 = Z ** 3
%timeit Z3 = np.power(Z, 3)
# enisum:  Evaluates the Einstein summation convention on the operands.
%timeit Z4 = np.einsum('i,i,i->i',Z,Z,Z)

```

    100 loops, best of 3: 2.16 ms per loop
    100 loops, best of 3: 11.1 ms per loop
    100 loops, best of 3: 11.5 ms per loop
    100 loops, best of 3: 2.66 ms per loop
    

#### 93. Consider two arrays A and B of shape (8,3) and (2,2). How to find rows of A that contain elements of each row of B regardless of the order of the elements in B? (★★★)


```python
A = np.random.randint(0, 10, (8, 3))
B = np.random.randint(0, 10, (2, 2))
print A
print B
C = (A[..., np.newaxis, np.newaxis] == B)
rows = np.where(C.any((3,1)).all(1))[0]
print(rows)
```

    [[0 9 0]
     [6 1 1]
     [3 6 9]
     [1 0 8]
     [3 7 7]
     [1 6 3]
     [1 1 8]
     [5 8 6]]
    [[1 9]
     [5 1]]
    [1 3 5 6]
    

#### 94. Considering a 10x3 matrix, extract rows with unequal values (e.g. \[2,2,3\]) (★★★)


```python
Z = np.random.randint(0, 2, (10, 3))
# print Z
uni_size = np.asarray([len(np.unique(Z[row])) for row in range(Z.shape[0])])
print Z[np.nonzero(uni_size>1)[0],:]

# # solution for arrays of all dtypes (including string arrays and record arrays)
# E = np.all(Z[:,1:] == Z[:,:-1], axis=1)
# U = Z[~E]
# print(U)
# # soluiton for numerical arrays only, will work for any number of columns in Z
# U = Z[Z.max(axis=1) != Z.min(axis=1),:]
# print(U)

```

    [[0 1 0]
     [0 1 1]
     [1 1 0]
     [0 0 1]
     [1 1 0]
     [0 0 0]
     [0 0 1]
     [0 0 0]
     [1 0 1]
     [1 1 0]]
    [[0 1 0]
     [0 1 1]
     [1 1 0]
     [0 0 1]
     [1 1 0]
     [0 0 1]
     [1 0 1]
     [1 1 0]]
    

#### 95. Convert a vector of ints into a matrix binary representation (★★★)


```python
# Author: Warren Weckesser

I = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128])
B = ((I.reshape(-1,1) & (2**np.arange(8))) != 0).astype(int)
print(B[:,::-1])

# Author: Daniel T. McDonald
# unpackbit: Unpacks elements of a uint8 array into a binary-valued output array.
I = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128], dtype=np.uint8)
print(np.unpackbits(I[:, np.newaxis], axis=1))
```

    [[0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 1]
     [0 0 0 0 0 0 1 0]
     [0 0 0 0 0 0 1 1]
     [0 0 0 0 1 1 1 1]
     [0 0 0 1 0 0 0 0]
     [0 0 1 0 0 0 0 0]
     [0 1 0 0 0 0 0 0]
     [1 0 0 0 0 0 0 0]]
    [[0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 1]
     [0 0 0 0 0 0 1 0]
     [0 0 0 0 0 0 1 1]
     [0 0 0 0 1 1 1 1]
     [0 0 0 1 0 0 0 0]
     [0 0 1 0 0 0 0 0]
     [0 1 0 0 0 0 0 0]
     [1 0 0 0 0 0 0 0]]
    

#### 96. Given a two dimensional array, how to extract unique rows? (★★★)


```python
Z = np.random.randint(0, 2, (10, 2))
# np.ascontiguousarray: Return a contiguous array in memory (C order).
T = np.ascontiguousarray(Z).view(np.dtype((np.void, Z.dtype.itemsize * Z.shape[1])))
_, idx = np.unique(T, return_index=True)
uZ = Z[idx]
print(uZ)
```

    [[0 0]
     [0 1]
     [1 0]]
    

#### 97. Considering 2 vectors A & B, write the einsum equivalent of inner, outer, sum, and mul function (★★★)


```python
A = np.random.uniform(0,1,10)
B = np.random.uniform(0,1,10)

np.einsum('i,i', A, B) # inner
np.einsum('i,j->ij', A, B) #outer
np.einsum('i->', A) # sum
np.einsum('i,i -> i', A, B) # mul

```




    array([ 0.33659014,  0.3989222 ,  0.142318  ,  0.17536192,  0.26035083,
            0.05752125,  0.00777259,  0.0017277 ,  0.06871543,  0.35018037])



#### 98. Considering a path described by two vectors (X,Y), how to sample it using equidistant samples (★★★)?


```python
# numpy.interp(x, xp, fp, left=None, right=None, period=None)
# One-dimensional linear interpolation.
# xp and fp is the given plot, the x is the point to interpolate,return the corresponding y
X = [1, 2, 3, 4, 5, 6]
Y = [9, 8, 7, 6, 5, 4]
x = np.random.uniform(0, 10, 10)
print x
print np.interp(x, X, Y)
```

    [ 4.44628383  5.82283132  2.49989208  2.51324338  3.25249002  3.68690241
      5.08205922  7.10483106  0.39595691  3.96266892]
    [ 5.55371617  4.17716868  7.50010792  7.48675662  6.74750998  6.31309759
      4.91794078  4.          9.          6.03733108]
    

#### 99. Given an integer n and a 2D array X, select from X the rows which can be interpreted as draws from a multinomial distribution with n degrees, i.e., the rows which only contain integers and which sum to n. (★★★)


```python
X = np.random.randint(0, 4, (10,3))
n = 4
M = np.logical_and.reduce(np.mod(X, 1) == 0, axis=1)
M &= (X.sum(axis=1) == n)
print M
print X[M,:]
```

    [False False False  True False  True False False False False]
    [[0 3 1]
     [2 0 2]]
    

#### 100. Compute bootstrapped 95% confidence intervals for the mean of a 1D array X (i.e., resample the elements of an array with replacement N times, compute the mean of each sample, and then compute percentiles over the means). (★★★)


```python
# 95% confidence intervals 置信度为95%的置信区间，均值的置信空间
# numpy.percentile(a, q, axis=None, out=None, overwrite_input=False, interpolation='linear', keepdims=False)
# Compute the qth percentage of the data along the specified axis.
X = np.random.randn(100) 
N = 1000 
idx = np.random.randint(0, X.size, (N, X.size))
means = X[idx].mean(axis=1) # 采样1000次，每次计算采样出来的样本的平均值
confint = np.percentile(means, [2.5, 97.5]) # 置信空间左右对称2.5%
print(confint)
```

    [-0.07589244  0.32095821]
    


```python

```
