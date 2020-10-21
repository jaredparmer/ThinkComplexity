""" fiddling with NumPy arrays to learn about them

see: https://jakevdp.github.io/PythonDataScienceHandbook/
02.02-the-basics-of-numpy-arrays.html

"""

import numpy as np

np.random.seed(0)

x1 = np.random.randint(10, size=6) # one-dimensional array
x2 = np.random.randint(10, size=(3, 4)) # two-dimensional array
x3 = np.random.randint(10, size=(3, 4, 5)) # three-dimensional array

""" array attributes:
    ndim, number of dimensions;
    shape, size of each dimension;
    size, size of total array
    dtype, data type of array
    itemsize, size in bytes of each array element
    nbytes, size in bytes of total array
"""

print("x1: ", x1)
print("x2: ", x2)
print("x3: ", x3)

print("x3 ndim:  ", x3.ndim)
print("x3 shape: ", x3.shape)
print("x3 size:  ", x3.size)
print("x3 dtype: ", x3.dtype)
print("x3 itemsize: ", x3.itemsize)
print("x3 nbytes: ", x3.nbytes)
print(f"nbytes == itemsize * size: {x3.nbytes == x3.itemsize * x3.size}")

# access by index
print("x1[2]: ", x1[2])
print("x1[-1]: ", x1[-1]) # access last element
print("x3[0, 2, 4]: ", x3[0, 2, 4])
print("x3[0][2][4]: ", x3[0][2][4])
""" also works, but probably a bad habit because this style does not work when
slicing! """

# modification by index
x3[0][2][4] = 1000000
print("assignment: x3[0][2][4] = 1000000")
print("x3: ", x3)

# dtype is fixed, so ValueErrors will get thrown. e.g.:
# x2[0][0] = 'hello'

# and truncation can happen
x2[0][0] = 3.14
print("assignment: x2[0][0] = 3.14")
print(f"x2[0][0] == 3.14: {x2[0][0] == 3.14}")
print(f"x2[0][0] == 3: {x2[0][0] == 3}")

# slicing one-dimensional arrays is just like slicing lists

# slicing multi-dimensional arrays:
print("x2: ", x2)
print("x2[:2, :3]: ", x2[:2, :3])   # works
print("x2[:2][:3]: ", x2[:2][:3])   # DOES NOT WORK!

# accessing entire rows or columns
print("x2[:, 0]: ", x2[:, 0]) # first column
print("x2[0, :]: ", x2[0, :]) # first row

""" IMPORTANT: slicing arrays return *views*, not *copies* of original arrays.
So modifying a slice of an array will modify the original array! """
x2_sub = x2[:2, :2]
x2_sub[0, 0] = 99
print("assignment: x2_sub[0, 0] = 99")
print(f"x2_sub[0, 0] == x2[0, 0]: {x2_sub[0, 0] == x2[0, 0]}")

# copying arrays is done with copy(). e.g.:
x2_sub = x2[:2, :2].copy()
x2_sub[0, 0] = 0
print("assignment after copy(): x2_sub[0, 0] = 0")
print(f"x2_sub[0, 0] == x2[0, 0]: {x2_sub[0, 0] == x2[0, 0]}")

# reshaping arrays
""" NumPy has a reshape() function for this, which will usually return a
no-copy view of the original array """
print("x1:\n", x1)
print("x1.reshape(2, 3):\n", x1.reshape(2, 3))

""" the following throws a ValueError because there aren't enough elements
in the original array to reshape to the 2-row, 4-column specification """
# print("x1.reshape(2, 4):\n", x1.reshape(2, 4))

# the NumPy keyword 'newaxis' can be used for reshaping by row or column
print("x1[:, np.newaxis]:\n", x1[:, np.newaxis]) # column vector
print("x1[np.newaxis, :]:\n", x1[np.newaxis, :])
""" IMPORTANT: using newaxis in this last instance creates a row vector, which
functions like a two-dimensional array and has to be accessed accordingly. For
example, while x1[3] gives you the fourth element in that one-dimensional
array, x1[np.newaxis, :][3] will throw a ValueError. Semantically, you need to
use x1[np.newaxis, :][0, 3]. """

""" using reshape() will also create vectors. So, for example, x1.reshape(6, 1)
will turn the one-dimensional, six-element array x1 into a six-row, one-column
two-dimensional array; and reshaping THAT back into a row produces a row vector
with six elements. So, e.g.: """
col = x1.reshape(6, 1)
row = col.reshape(1, 6)
print("col = x1.reshape(6, 1):\n", col)
print("row = x1.reshape(6, 1).reshape(1, 6):\n", row)

# row[1] will NOT give you x1[1], but throw a ValueError
# row[0][1] will, however, give you x1[1]
print(f"x1[1] == row[0][1]: {x1[1] == row[0][1]}")
try:
    print(f"x1[1] == row[1]: {x1[1] == row[1]}")
except:
    print("x1[1] == row[1] gives ValueError!")

# concatenating arrays
# can always use concatenate():
x = np.array([1, 2, 3])
y = np.array([3, 2, 1])
xy_h = np.concatenate([x, y])
print("xy_h:\n", xy_h)
z = np.array([99, 99, 99])
# concatenation of multiple arrays possible
xyz_h = np.concatenate([x, y, z])
print("xyz_h:\n", xyz_h)

# concatenate() works on multi-dimensional arrays, with axis default of 0
grid = np.array([[1, 2, 3],
                [4, 5, 6]])
print("grid:\n", grid)
print("np.concatenate([grid, grid]):\n", np.concatenate([grid, grid]))
print("np.concatenate([grid, grid], axis=1):\n",
      np.concatenate([grid, grid], axis=1))
