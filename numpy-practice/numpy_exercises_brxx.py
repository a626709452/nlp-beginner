
# coding: utf-8

# # 100 numpy exercises
# 
# This is a collection of exercises that have been collected in the numpy mailing list, on stack overflow and in the numpy documentation. The goal of this collection is to offer a quick reference for both old and new users but also to provide a set of exercices for those who teach.
# 
# 
# If you find an error or think you've a better way to solve some of them, feel free to open an issue at <https://github.com/rougier/numpy-100>

# #### 1. Import the numpy package under the name `np` (★☆☆)

# In[1]:

import numpy as np


# #### 2. Print the numpy version and the configuration (★☆☆)

# In[3]:

print np.__version__
print np.version.version
np.show_config()


# #### 3. Create a null vector of size 10 (★☆☆)

# In[5]:

a = np.zeros(10)


# #### 4.  How to find the memory size of any array (★☆☆)

# In[8]:

print a.size
print a.itemsize
print a.size * a.itemsize


# #### 5.  How to get the documentation of the numpy add function from the command line? (★☆☆)

# In[15]:

get_ipython().system(u'python -c "import numpy;numpy.info(numpy.add)"')
# ! for command line
# python -c "" for python command line


# #### 6.  Create a null vector of size 10 but the fifth value which is 1 (★☆☆)

# In[17]:

b = np.zeros(10)
b[4] = 1
print b


# #### 7.  Create a vector with values ranging from 10 to 49 (★☆☆)

# In[25]:

c = np.arange(10, 49)
print c


# #### 8.  Reverse a vector (first element becomes last) (★☆☆)

# In[32]:

d = np.arange(10, 49)
print d[::-1]


# #### 9.  Create a 3x3 matrix with values ranging from 0 to 8 (★☆☆)

# In[36]:

print np.arange(0, 9).reshape(3, 3)


# #### 10. Find indices of non-zero elements from \[1,2,0,0,4,0\] (★☆☆)

# In[48]:

a = np.array([1,2,0,0,4,0])
print np.arange(0, a.size)[a!=0]
# nonzero, return tuple: the first element means the axis=0 of the indice and the second element means the axis=1 of the indice
# eg. if the a[0,0], a[1,2], a[3,4] is nonzero, then the function returns (array([0,1,3]), array(0,2,4)) 
print np.nonzero([1,2,0,0,4,0])[0]


# #### 11. Create a 3x3 identity matrix (★☆☆)

# In[49]:

np.eye(3,3)


# #### 12. Create a 3x3x3 array with random values (★☆☆)

# In[51]:

np.random.random([3,3,3])


# #### 13. Create a 10x10 array with random values and find the minimum and maximum values (★☆☆)

# In[55]:

a = np.random.random([10,10])
print a
print a.min(), a.max()


# #### 14. Create a random vector of size 30 and find the mean value (★☆☆)

# In[56]:

a = np.random.random(30)
print a.mean()


# #### 15. Create a 2d array with 1 on the border and 0 inside (★☆☆)

# In[64]:

a = np.zeros([10,10])
a[[0,-1],:] = 1
a[:,[0,-1]] = 1
print a


# #### 16. How to add a border (filled with 0's) around an existing array? (★☆☆)

# In[75]:

a = np.random.random([3,3])
# print np.pad(a, pad_width=((3,2),(1,4)), mode='maximum')
# 3 on the top and 2 on the bottom, 1 on the left and 4 on the right
print np.pad(a,pad_width=1,mode='constant',constant_values=0)


# #### 17. What is the result of the following expression? (★☆☆)

# ```python
# 0 * np.nan
# np.nan == np.nan
# np.inf > np.nan
# np.nan - np.nan
# 0.3 == 3 * 0.1
# ```

# In[76]:

# any number operated with nan will get nan
# nan can not be compared
print 0 * np.nan # nan
print np.nan == np.nan # False
print np.inf > np.nan # False
print np.nan - np.nan # nan
print 0.3 == 3 * 0.1 # False


# #### 18. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal (★☆☆)

# In[79]:

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


# #### 19. Create a 8x8 matrix and fill it with a checkerboard pattern (★☆☆)

# In[95]:

a = np.zeros([8,8])
a[0::2,1::2] = 1
a[1::2, 0::2] = 1
print a


# #### 20. Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element?

# In[102]:

a = np.arange(0,6*7*8).reshape([6,7,8])
print np.nonzero(a==99) # a[1,5,3]


# #### 21. Create a checkerboard 8x8 matrix using the tile function (★☆☆)

# In[103]:

# tile function refers to https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.tile.html
a = [[0,1],[1,0]]
print np.tile(a, [4,4])


# #### 22. Normalize a 5x5 random matrix (★☆☆)

# In[109]:

a = np.random.random([5,5])
mean = a.mean()
var = b.std()
a = (a - mean) / var
print a
# vector=(vector-vector.min())/(vector.max()-vector.min())


# #### 23. Create a custom dtype that describes a color as four unsigned bytes (RGBA) (★☆☆)

# In[114]:

# dtype refers to https://docs.scipy.org/doc/numpy/reference/generated/numpy.dtype.html
# the difference between astype and dtype can refer to http://www.mamicode.com/info-detail-1180317.html
color = np.dtype([('R',np.ubyte), ('G', np.ubyte), ('B', np.ubyte), ('A', np.ubyte)])
print color


# #### 24. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product) (★☆☆)

# In[122]:

a = np.random.random([5, 3])
b = np.random.random([3, 2])
print np.dot(a, b)


# #### 25. Given a 1D array, negate all elements which are between 3 and 8, in place. (★☆☆)

# In[134]:

a = np.arange(1, 10)
a[(a > 3) & (a < 8)] *= -1
print a


# #### 26. What is the output of the following script? (★☆☆)

# ```python
# # Author: Jake VanderPlas
# 
# print(sum(range(5),-1))
# from numpy import *
# print(sum(range(5),-1))
# ```

# In[140]:

print(sum(range(5),-1)) # 9,sum is the build-in function in python，sum(sequence[,start]) equal to start+1+2+3+4=9
print(np.sum(range(5),-1)) #numpy.sum(a, axis=None)


# #### 27. Consider an integer vector Z, which of these expressions are legal? (★☆☆)

# ```python
# Z**Z
# 2 << Z >> 2
# Z <- Z
# 1j*Z
# Z/1/1
# Z<Z>Z
# ```

# In[160]:

Z = np.arange(0,5)
print Z
print Z ** Z # [0^0, 1^1, 2^2, 3^3, 4^4]
print 2 << Z >> 2
print Z <- Z # -> refer to the return type of the function
print 1j*Z
print Z/1/1
# print Z<Z>Z


# #### 28. What are the result of the following expressions?

# ```python
# np.array(0) / np.array(0)
# np.array(0) // np.array(0)
# np.array([np.nan]).astype(int).astype(float)
# ```

# In[169]:

from __future__ import division
# print np.array(0) / np.array(0) # nan, runtime warning (float division)
# print np.array(0) // np.array(0) #runtime warning (int division)
np.array([np.nan]).astype(int).astype(float)


# #### 29. How to round away from zero a float array ? (★☆☆)

# In[173]:

a = np.random.uniform(-10, 10, 10)
print a
# copysign for copysign from Z
print (np.copysign(np.ceil(np.abs(a)), a))


# #### 30. How to find common values between two arrays? (★☆☆)

# In[174]:

a = np.random.randint(0, 10, 10)
b = np.random.randint(0, 10, 10)
print a
print b
print(np.intersect1d(a, b))


# #### 31. How to ignore all numpy warnings (not recommended)? (★☆☆)

# In[175]:

# Suicide mode on
defaults = np.seterr(all="ignore")
Z = np.ones(1) / 0

# Back to sanity
_ = np.seterr(**defaults)

#An equivalent way, with a context manager:
with np.errstate(divide='ignore'):
    Z = np.ones(1) / 0


# #### 32. Is the following expressions true? (★☆☆)

# ```python
# np.sqrt(-1) == np.emath.sqrt(-1)
# ```

# In[178]:

# print np.sqrt(-1) # nan
print np.emath.sqrt(-1) # 1j


# #### 33. How to get the dates of yesterday, today and tomorrow? (★☆☆)

# In[182]:

print np.datetime64('today', 'D') - np.timedelta64(1, 'D')
print np.datetime64('today', 'D')
print np.datetime64('today', 'D') + np.timedelta64(1, 'D')


# #### 34. How to get all the dates corresponding to the month of July 2016? (★★☆)

# In[183]:

print np.arange('2016-07', '2016-08', dtype='datetime64[D]')


# #### 35. How to compute ((A+B)\*(-A/2)) in place (without copy)? (★★☆)

# In[190]:

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


# #### 36. Extract the integer part of a random array using 5 different methods (★★☆)

# In[201]:

a = np.random.uniform(0, 10, 10)
print a
print a.astype(int)
print np.floor(a)
print np.ceil(a) - 1
print a - a % 1
print np.trunc(a) # tail-cut


# #### 37. Create a 5x5 matrix with row values ranging from 0 to 4 (★★☆)

# In[206]:

Z = np.zeros((5, 5))
Z += np.arange(5)
print(Z)
a = np.arange(5)
print np.tile(a, [5,1])


# #### 38. Consider a generator function that generates 10 integers and use it to build an array (★☆☆)

# In[209]:

def generator():
    for x in range(0, 10):
        yield x
# numpy.fromiter ---- build an ndarray object from an iterable object
# numpy.fromiter(iterable, dtype, count = -1)
# count : int, optional The number of items to read from iterable. The default is -1, which means all data is read.
Z = np.fromiter(generator(),dtype=int,count=-1)
print Z


# #### 39. Create a vector of size 10 with values ranging from 0 to 1, both excluded (★★☆)

# In[212]:

print np.linspace(0, 1, 11, endpoint=False)[1:]
# endpoint : bool, optional If True, stop is the last sample. Otherwise, it is not included. Default is True.


# #### 40. Create a random vector of size 10 and sort it (★★☆)

# In[216]:

a = np.random.random(10)
print a
a.sort()
print a


# #### 41. How to sum a small array faster than np.sum? (★★☆)

# In[3]:

a = np.random.random(10)
print a
print np.add.reduce(a)


# #### 42. Consider two random array A and B, check if they are equal (★★☆)

# In[5]:

A = np.random.random(10)
B = np.random.random(10)
# True if two arrays have the same shape and elements, False otherwise.
print np.array_equal(A,B)
# Returns True if two arrays are element-wise equal within a tolerance.
print np.allclose(A, B)


# #### 43. Make an array immutable (read-only) (★★☆)

# In[10]:

a = np.eye(3)
print a
print a.flags
a.setflags(write=0)


# #### 44. Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates (★★☆)

# In[28]:

a = np.random.randint(1,100,10*2).reshape((10,2))
print a
b = np.arctan2(a[:,0], a[:,1])
c = np.sqrt(a[:,0]**2 + a[:,1]**2)
print zip(b, c)


# #### 45. Create random vector of size 10 and replace the maximum value by 0 (★★☆)

# In[32]:

a = np.random.randint(1,100,10)
print a
a[a==a.max()] = 0
print a
a[a.argmax()]=0
print a


# #### 46. Create a structured array with `x` and `y` coordinates covering the \[0,1\]x\[0,1\] area (★★☆)

# In[33]:

# Return coordinate matrices from coordinate vectors.
print np.meshgrid([0,1],[0,1])


# ####  47. Given two arrays, X and Y, construct the Cauchy matrix C (Cij =1/(xi - yj))

# In[39]:

X = np.random.randint(1,100,3)
Y = np.random.randint(1,100,5)
print X
print Y
from __future__ import division
print 1 / np.subtract.outer(X,Y)


# #### 48. Print the minimum and maximum representable value for each numpy scalar type (★★☆)

# In[46]:

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


# #### 49. How to print all the values of an array? (★★☆)

# In[49]:

# Set printing options
# threshold, Total number of array elements which trigger summarization 
#            rather than full repr (default 1000).
np.set_printoptions(threshold=np.nan)
Z = np.zeros((16,16))
print(Z)


# #### 50. How to find the closest value (to a given scalar) in a vector? (★★☆)

# In[50]:

a = np.random.randint(0,100,10)
x = 50
print a
temp = np.abs(a - x)
print a[temp.argmin()]


# #### 51. Create a structured array representing a position (x,y) and a color (r,g,b) (★★☆)

# In[73]:

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


# #### 52. Consider a random vector with shape (100,2) representing coordinates, find point by point distances (★★☆)

# In[88]:

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


# #### 53. How to convert a float (32 bits) array into an integer (32 bits) in place?

# In[102]:

a = np.arange(10, dtype=np.float32)
print a.dtype
a = a.astype(np.int32, copy=0)
print a.dtype


# #### 54. How to read the following file? (★★☆)

# ```
# 1, 2, 3, 4, 5
# 6,  ,  , 7, 8
#  ,  , 9,10,11
# ```

# In[104]:

# numpy.genfromtxt: Load data from a text file, with missing values handled as specified.
# refer to: https://docs.scipy.org/doc/numpy/reference/generated/numpy.genfromtxt.html
a = np.genfromtxt('input1.txt', delimiter=',')
print a


# #### 55. What is the equivalent of enumerate for numpy arrays? (★★☆)

# In[108]:

# enumerate in python can get index and value at the same time
Z = np.arange(9).reshape(3,3)
for index, value in np.ndenumerate(Z):
    print index, value 
# for index in np.ndindex(Z.shape):
#     print index, Z[index]


# #### 56. Generate a generic 2D Gaussian-like array (★★☆)

# In[4]:

X,Y = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
# print X
# print Y
D = np.sqrt(X * X + Y * Y)
sigma, mu = 1.0, 0.0
G = np.exp(-((D - mu) ** 2 / (2.0 * sigma ** 2)))
print G


# #### 57. How to randomly place p elements in a 2D array? (★★☆)

# In[7]:

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


# #### 58. Subtract the mean of each row of a matrix (★★☆)

# In[18]:

a = np.random.randint(0,100, 5*8).reshape(5, 8)
print a
mean = a.mean(axis=1, keepdims=True)
print a - mean


# #### 59. How to sort an array by the nth column? (★★☆)

# In[23]:

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


# #### 60. How to tell if a given 2D array has null columns? (★★☆)

# In[32]:

a = np.random.randint(0, 3, (3, 10))
print a
print a.any(axis=0)
print ((~a.any(axis=0)).any())


# #### 61. Find the nearest value from a given value in an array (★★☆)

# In[34]:

a = np.random.randint(0, 100, 10)
print a
x = 6
res = a[np.argmin(np.abs(a-x))]
print res


# #### 62. Considering two arrays with shape (1,3) and (3,1), how to compute their sum using an iterator? (★★☆)

# In[51]:

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


# #### 63. Create an array class that has a name attribute (★★☆)

# In[52]:

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


# #### 64. Consider a given vector, how to add 1 to each element indexed by a second vector (be careful with repeated indices)? (★★★)

# In[58]:

a = np.zeros(10)
idx = [2, 5, 9, 2]
a[idx] += 1
print a
np.add.at(a, idx, 1)
print(a)


# #### 65. How to accumulate elements of a vector (X) to an array (F) based on an index list (I)? (★★★)

# In[64]:

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


# #### 66. Considering a (w,h,3) image of (dtype=ubyte), compute the number of unique colors (★★★)

# In[74]:

I = np.random.randint(0, 2, (5,5,3)).astype(np.ubyte)
F = I[...,0]*256*256 + I[...,1]*256 +I[...,2]
print F
n = len(np.unique(F))
print(n)


# #### 67. Considering a four dimensions array, how to get sum over the last two axis at once? (★★★)

# In[77]:

a = np.random.randint(0, 100, (5,5,2,3))
# print a
print a.sum(axis=(-2,-1))


# #### 68. Considering a one-dimensional vector D, how to compute means of subsets of D using a vector S of same size describing subset  indices? (★★★)

# In[83]:

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


# #### 69. How to get the diagonal of a dot product? (★★★)

# In[86]:

A = np.random.uniform(0,1,(5,5))
B = np.random.uniform(0,1,(5,5))
print np.diag(np.dot(A, B))
# Fast version
print np.sum(A * B.T, axis=1)
# Faster version
print np.einsum("ij,ji->i", A, B)


# #### 70. Consider the vector \[1, 2, 3, 4, 5\], how to build a new vector with 3 consecutive zeros interleaved between each value? (★★★)

# In[89]:

Z = np.array([1,2,3,4,5])
nz = 3
Z0 = np.zeros(len(Z) + (len(Z)-1)*(nz))
Z0[::nz+1] = Z
print(Z0)


# #### 71. Consider an array of dimension (5,5,3), how to mulitply it by an array with dimensions (5,5)? (★★★)

# In[8]:

A = np.ones((5,5,3))
B = 2*np.ones((5,5))
# add one dim
print B[:,:,None].shape #(5, 5, 1)
print (A * B[:,:,None]).shape #(5, 5, 3)


# #### 72. How to swap two rows of an array? (★★★)

# In[17]:

A = np.arange(25).reshape(5,5)
print A
# A[(0,1)] = A[(1,0)] () refer to the elements, this doesn't swap but copy
A[:,[0,1]] = A[:,[1,0]] # [] refer to the row
print A


# #### 73. Consider a set of 10 triplets describing 10 triangles (with shared vertices), find the set of unique line segments composing all the  triangles (★★★)

# In[9]:

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


# #### 74. Given an array C that is a bincount, how to produce an array A such that np.bincount(A) == C? (★★★)

# In[11]:

C = np.random.randint(0, 10, 10)
print C
A = np.repeat(np.arange(len(C)), C)
print A


# #### 75. How to compute averages using a sliding window over an array? (★★★)

# In[18]:

from __future__ import division
def mv_avg(a, n = 3):
    ret = np.cumsum(a, dtype=float)
    print ret
    ret[n:] = ret[n:] - ret[:-n]
    print ret
    return ret[n-1:]/n
Z = np.arange(20)
print mv_avg(Z)


# #### 76. Consider a one-dimensional array Z, build a two-dimensional array whose first row is (Z\[0\],Z\[1\],Z\[2\]) and each subsequent row is  shifted by 1 (last row should be (Z\[-3\],Z\[-2\],Z\[-1\]) (★★★)

# In[44]:

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


# #### 77. How to negate a boolean, or to change the sign of a float inplace? (★★★)

# In[49]:

flag = np.random.randint(0, 2, 100)
print np.logical_not(flag, out = flag)
f = np.random.uniform(0, 1, 100)
print np.negative(f, out = f)


# #### 78. Consider 2 sets of points P0,P1 describing lines (2d) and a point p, how to compute distance from p to each line i  (P0\[i\],P1\[i\])? (★★★)

# In[32]:

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


# #### 79. Consider 2 sets of points P0,P1 describing lines (2d) and a set of points P, how to compute distance from each point j (P\[j\]) to each line i (P0\[i\],P1\[i\])? (★★★)

# In[40]:

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


# #### 80. Consider an arbitrary array, write a function that extract a subpart with a fixed shape and centered on a given element (pad with a `fill` value when necessary) (★★★)

# In[ ]:

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


# #### 81. Consider an array Z = \[1,2,3,4,5,6,7,8,9,10,11,12,13,14\], how to generate an array R = \[\[1,2,3,4\], \[2,3,4,5\], \[3,4,5,6\], ..., \[11,12,13,14\]\]? (★★★)

# In[44]:

from numpy.lib import stride_tricks
# numpy.lib.stride_tricks.as_strided(x, shape=None, strides=None, subok=False, writeable=True)
## strides: The strides of the new array, strides = (a.itemsize, a.itemsize)
Z = np.arange(1,15,dtype=np.uint32)
R = stride_tricks.as_strided(Z,(11,4),(4,4))
print(R)


# #### 82. Compute a matrix rank (★★★)

# In[49]:

A = np.random.randint(0, 100, (7, 8))
print A
U, S, V = np.linalg.svd(A)
rank = np.sum(S > 1e-10)
print(rank)
print np.linalg.matrix_rank(A)


# #### 83. How to find the most frequent value in an array?

# In[51]:

Z = np.random.randint(0, 10, 100)
print Z
C = np.bincount(Z)
print C
print np.argmax(C)


# #### 84. Extract all the contiguous 3x3 blocks from a random 10x10 matrix (★★★)

# In[55]:

Z = np.random.randint(0, 10, (10, 10), dtype=np.int32)
print Z.strides # (40, 4) offset = sum(np.array(i) * a.strides)
print Z.strides + Z.strides #(40, 4, 40, 4)
print stride_tricks.as_strided(Z, shape=(8, 8, 3, 3), strides=Z.strides + Z.strides)


# #### 85. Create a 2D array subclass such that Z\[i,j\] == Z\[j,i\] (★★★)

# In[64]:

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


# #### 86. Consider a set of p matrices wich shape (n,n) and a set of p vectors with shape (n,1). How to compute the sum of of the p matrix products at once? (result has shape (n,1)) (★★★)

# In[58]:

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


# #### 87. Consider a 16x16 array, how to get the block-sum (block size is 4x4)? (★★★)

# In[71]:

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


# #### 88. How to implement the Game of Life using numpy arrays? (★★★)

# In[ ]:

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


# #### 89. How to get the n largest values of an array (★★★)

# In[2]:

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


# #### 90. Given an arbitrary number of vectors, build the cartesian product (every combinations of every item) (★★★)

# In[11]:

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


# #### 91. How to create a record array from a regular array? (★★★)

# In[13]:

# core.records.fromarrays: create a record array from a (flat) list of arrays
Z = np.array([("Hello", 2.5, 3),
              ("World", 3.6, 2)])
R = np.core.records.fromarrays(Z.T, 
                               names='col1, col2, col3',
                               formats = 'S8, f8, i8')
print(R)
print R.col1
print R[0]


# #### 92. Consider a large vector Z, compute Z to the power of 3 using 3 different methods (★★★)

# In[14]:

Z = np.random.random(100000)
get_ipython().magic(u'timeit Z1 = Z * Z * Z')
get_ipython().magic(u'timeit Z2 = Z ** 3')
get_ipython().magic(u'timeit Z3 = np.power(Z, 3)')
# enisum:  Evaluates the Einstein summation convention on the operands.
get_ipython().magic(u"timeit Z4 = np.einsum('i,i,i->i',Z,Z,Z)")


# #### 93. Consider two arrays A and B of shape (8,3) and (2,2). How to find rows of A that contain elements of each row of B regardless of the order of the elements in B? (★★★)

# In[18]:

A = np.random.randint(0, 10, (8, 3))
B = np.random.randint(0, 10, (2, 2))
print A
print B
C = (A[..., np.newaxis, np.newaxis] == B)
rows = np.where(C.any((3,1)).all(1))[0]
print(rows)


# #### 94. Considering a 10x3 matrix, extract rows with unequal values (e.g. \[2,2,3\]) (★★★)

# In[37]:

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


# #### 95. Convert a vector of ints into a matrix binary representation (★★★)

# In[40]:

# Author: Warren Weckesser

I = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128])
B = ((I.reshape(-1,1) & (2**np.arange(8))) != 0).astype(int)
print(B[:,::-1])

# Author: Daniel T. McDonald
# unpackbit: Unpacks elements of a uint8 array into a binary-valued output array.
I = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128], dtype=np.uint8)
print(np.unpackbits(I[:, np.newaxis], axis=1))


# #### 96. Given a two dimensional array, how to extract unique rows? (★★★)

# In[44]:

Z = np.random.randint(0, 2, (10, 2))
# np.ascontiguousarray: Return a contiguous array in memory (C order).
T = np.ascontiguousarray(Z).view(np.dtype((np.void, Z.dtype.itemsize * Z.shape[1])))
_, idx = np.unique(T, return_index=True)
uZ = Z[idx]
print(uZ)


# #### 97. Considering 2 vectors A & B, write the einsum equivalent of inner, outer, sum, and mul function (★★★)

# In[48]:

A = np.random.uniform(0,1,10)
B = np.random.uniform(0,1,10)

np.einsum('i,i', A, B) # inner
np.einsum('i,j->ij', A, B) #outer
np.einsum('i->', A) # sum
np.einsum('i,i -> i', A, B) # mul


# #### 98. Considering a path described by two vectors (X,Y), how to sample it using equidistant samples (★★★)?

# In[52]:

# numpy.interp(x, xp, fp, left=None, right=None, period=None)
# One-dimensional linear interpolation.
# xp and fp is the given plot, the x is the point to interpolate,return the corresponding y
X = [1, 2, 3, 4, 5, 6]
Y = [9, 8, 7, 6, 5, 4]
x = np.random.uniform(0, 10, 10)
print x
print np.interp(x, X, Y)


# #### 99. Given an integer n and a 2D array X, select from X the rows which can be interpreted as draws from a multinomial distribution with n degrees, i.e., the rows which only contain integers and which sum to n. (★★★)

# In[59]:

X = np.random.randint(0, 4, (10,3))
n = 4
M = np.logical_and.reduce(np.mod(X, 1) == 0, axis=1)
M &= (X.sum(axis=1) == n)
print M
print X[M,:]


# #### 100. Compute bootstrapped 95% confidence intervals for the mean of a 1D array X (i.e., resample the elements of an array with replacement N times, compute the mean of each sample, and then compute percentiles over the means). (★★★)

# In[63]:

# 95% confidence intervals 置信度为95%的置信区间，均值的置信空间
# numpy.percentile(a, q, axis=None, out=None, overwrite_input=False, interpolation='linear', keepdims=False)
# Compute the qth percentage of the data along the specified axis.
X = np.random.randn(100) 
N = 1000 
idx = np.random.randint(0, X.size, (N, X.size))
means = X[idx].mean(axis=1) # 采样1000次，每次计算采样出来的样本的平均值
confint = np.percentile(means, [2.5, 97.5]) # 置信空间左右对称2.5%
print(confint)


# In[ ]:



