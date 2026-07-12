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
