# NumPy Notes

## 📌 What is NumPy?

NumPy (Numerical Python) is an open-source Python library used for numerical computing. It provides support for fast and memory-efficient multidimensional arrays along with a large collection of mathematical functions.

---

# 🚀 Why NumPy?

- Faster than Python Lists
- Memory Efficient
- Easy Mathematical Operations
- Supports Multi-dimensional Arrays
- Foundation of Machine Learning, Data Science & Deep Learning

---

# 📦 Array Creation

### Create Array

```python
import numpy as np

arr = np.array([1,2,3,4,5])
```

### arange()

```python
np.arange(1,11)
```

Creates equally spaced values within a range.

---

### linspace()

```python
np.linspace(-10,10,5)
```

Creates evenly spaced numbers between start and end.

---

### ones()

```python
np.ones((3,4))
```

Creates an array filled with ones.

---

### zeros()

```python
np.zeros((2,5))
```

Creates an array filled with zeros.

---

### identity()

```python
np.identity(4)
```

Creates an identity matrix.

---

### Random Array

```python
np.random.random((3,4))
```

Random values between 0 and 1.

---

# 📐 Array Properties

```python
a = np.arange(12)
```

### dtype

Returns data type.

```python
a.dtype
```

---

### size

Returns total number of elements.

```python
a.size
```

---

### ndim

Returns dimensions.

```python
a.ndim
```

---

### shape

Returns rows and columns.

```python
a.shape
```

---

### itemsize

Memory occupied by one element.

```python
a.itemsize
```

---

### astype()

Changes data type.

```python
a.astype(np.int16)
```

---

# 🔢 Reshaping Arrays

```python
a = np.arange(12).reshape(3,4)
```

Converts a 1D array into a 2D array.

---

# ➕ Mathematical Operations

```python
a**2
```

Square of every element.

```python
a+5
```

Addition

```python
a-2
```

Subtraction

```python
a*2
```

Multiplication

```python
a/2
```

Division

---

# 📊 Statistical Functions

```python
np.sum(a)
```

Returns total sum.

```python
np.mean(a)
```

Average.

```python
np.max(a)
```

Maximum value.

```python
np.min(a)
```

Minimum value.

```python
np.prod(a)
```

Product of all elements.

```python
np.median(a)
```

Median.

```python
np.var(a)
```

Variance.

```python
np.std(a)
```

Standard Deviation.

---

# 🎯 Comparison Operations

```python
a > 5
```

Returns Boolean array.

Example:

```
[False False False True True]
```

Useful for filtering data.

---

# 🧮 Matrix Multiplication

```python
a = np.arange(12).reshape(2,6)

b = np.arange(12,24).reshape(6,2)

np.dot(a,b)
```

Performs matrix multiplication.

---

# 📈 Data Visualization

```python
import matplotlib.pyplot as plt

x = np.linspace(-10,10,100)
y = np.sin(x)

plt.plot(x,y)
plt.show()
```

Used to visualize numerical data.

---

# 💡 Key Learnings

✅ NumPy Arrays

✅ Array Creation

✅ Array Properties

✅ Reshaping

✅ Mathematical Operations

✅ Statistical Functions

✅ Matrix Multiplication

✅ Boolean Comparisons

✅ Basic Data Visualization using Matplotlib

---

# 📌 Next Topics

- Indexing
- Slicing
- Fancy Indexing
- Boolean Masking
- Broadcasting
- Vectorization
- Iteration

# Broadcasting

Broadcasting is a NumPy feature that allows arithmetic operations on arrays of different shapes without explicitly reshaping them.

### Advantages
- Faster computations
- Less memory usage
- Eliminates unnecessary loops
- Makes code cleaner and more readable

### Example

```python
import numpy as np

arr = np.array([1, 2, 3])

print(arr + 5)
# Output: [6 7 8]
```

### Broadcasting Between Arrays

```python
a = np.array([[1], [2], [3]])
b = np.array([10, 20, 30])

print(a + b)
```

Output

```
[[11 21 31]
 [12 22 32]
 [13 23 33]]
```

---

# Vectorization

Vectorization means performing operations on an entire array at once instead of using Python loops.

### Advantages

- High Performance
- Cleaner Code
- Optimized Memory Usage
- Essential for Machine Learning

### Example

```python
arr = np.array([1,2,3,4,5])

print(arr * 2)
```

Output

```
[2 4 6 8 10]
```

---

# Universal Functions (ufuncs)

NumPy provides built-in mathematical functions that operate element-wise.

### Common Universal Functions

```python
np.sqrt()
np.sin()
np.cos()
np.tan()
np.log()
np.exp()
np.abs()
```

### Example

```python
arr = np.array([1,4,9,16])

print(np.sqrt(arr))
```

Output

```
[1. 2. 3. 4.]
```

---

# Boolean Filtering

Boolean conditions can be used to filter array elements.

Example

```python
arr = np.array([10,20,30,40,50])

print(arr[arr > 25])
```

Output

```
[30 40 50]
```

---

# Key Learnings

- Broadcasting performs operations on arrays with compatible shapes.
- Vectorization replaces loops with efficient array operations.
- Universal Functions perform fast mathematical computations.
- Boolean Filtering helps select data based on conditions.
- These concepts are widely used in Data Science, Machine Learning, and Deep Learning.

# Stacking

Stacking combines two or more arrays into a single array.

## Vertical Stacking

- Uses `np.vstack()`
- Combines arrays row-wise.

Example:

```python
np.vstack((a, b))
```

## Horizontal Stacking

- Uses `np.hstack()`
- Combines arrays column-wise.

Example:

```python
np.hstack((a, b))
```

---

# Splitting Arrays

Splitting divides a large array into smaller arrays.

Function:

```python
np.split(array, sections)
```

Example:

```python
arr = np.arange(1,13)

np.split(arr,3)
```

Output:

```
[array([1,2,3,4]),
 array([5,6,7,8]),
 array([9,10,11,12])]
```

---

# Sorting Arrays

Sorting arranges elements in ascending order.

Function:

```python
np.sort()
```

Example:

```python
marks = np.array([78,45,99,32,67])

np.sort(marks)
```

Output:

```
[32 45 67 78 99]
```

---

# Searching Elements

The `np.where()` function returns the index of elements that satisfy a condition.

Example:

```python
marks = np.array([78,45,99,32,67])

np.where(marks == 67)
```

Output:

```
(array([4]),)
```

---

# Unique Values

`np.unique()` returns all unique elements from an array.

Example:

```python
arr = np.array([1,2,2,3,4,4,5])

np.unique(arr)
```

Output:

```
[1 2 3 4 5]
```

---

# Count Non-Zero Elements

`np.count_nonzero()` counts all non-zero values.

Example:

```python
arr = np.array([0,1,2,0,3,0,4])

np.count_nonzero(arr)
```

Output:

```
4
```

---

# Key Learnings

- Combined arrays using vertical and horizontal stacking.
- Split arrays into multiple equal parts.
- Sorted arrays efficiently using `np.sort()`.
- Located elements using `np.where()`.
- Extracted unique values using `np.unique()`.
- Counted non-zero elements with `np.count_nonzero()`.
- These operations are commonly used in data preprocessing and machine learning workflows.
