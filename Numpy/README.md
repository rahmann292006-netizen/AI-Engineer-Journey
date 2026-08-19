# NumPy Learning Journey 🚀

This folder documents my complete NumPy learning journey through hands-on practice, detailed notes, and real coding exercises.

---

## 📚 Topics Covered

### **Foundation & Array Basics**
- Array Creation Methods (`np.array`, `arange`, `linspace`, `ones`, `zeros`, `identity`, random arrays)
- Array Properties (`dtype`, `size`, `ndim`, `shape`, `itemsize`, `astype`)
- Array Reshaping & Manipulation

### **Mathematical & Statistical Operations**
- Element-wise Arithmetic Operations (`+`, `-`, `*`, `/`, `**`)
- Statistical Functions (`sum`, `mean`, `max`, `min`, `prod`, `median`, `var`, `std`)
- Comparison Operations & Boolean Masking
- Universal Functions (ufuncs): `sqrt`, `sin`, `cos`, `tan`, `log`, `exp`, `abs`

### **Advanced Array Techniques**
- **Indexing** - Single element, row/column access, negative indexing
- **Slicing** - Row/column ranges, submatrix extraction
- **Boolean Masking** - Conditional filtering (`arr[arr > 5]`)
- **Fancy Indexing** - Integer array indexing for rows/columns
- **Broadcasting** - Operations on arrays of different shapes
- **Vectorization** - Replacing loops with optimized array operations

### **Array Combining & Transforming**
- **Stacking** - Vertical (`vstack`) & Horizontal (`hstack`)
- **Splitting** - `np.split` for dividing arrays
- **Sorting** - `np.sort` for ascending order
- **Searching** - `np.where` for conditional indices
- **Unique Values** - `np.unique` for deduplication
- **Count Non-Zero** - `np.count_nonzero` for sparse data analysis

### **Linear Algebra**
- Matrix Multiplication (`np.dot`, `@` operator)
- Matrix Transpose (`.T`)
- Determinant (`np.linalg.det`)
- Matrix Inverse (`np.linalg.inv`)
- Matrix Rank (`np.linalg.matrix_rank`)
- Trace (`np.trace`)
- Eigenvalues & Eigenvectors (`np.linalg.eig`)

---

## 📂 Files in This Folder

| File | Description |
|------|-------------|
| `numpy_hands_on.ipynb` | **Complete hands-on notebook** with 70+ code cells covering all topics above, including practice challenges and mini-projects |
| `notes.md` | **Comprehensive concept notes** with explanations, code examples, and key learnings for each topic |
| `Projects/` | *(Directory)* Additional mini-projects applying NumPy to real-world scenarios |

---

## 🎯 Learning Outcomes

After completing this journey, I can confidently:

- ✅ **Create and manipulate** NumPy arrays of any dimension
- ✅ **Perform mathematical and statistical** computations efficiently
- ✅ **Apply indexing, slicing, and boolean masking** for precise data selection
- ✅ **Use broadcasting and vectorization** to write faster, cleaner code
- ✅ **Work with sorting, searching, stacking, and splitting** operations
- ✅ **Solve linear algebra problems** including matrix operations, decompositions, and eigenvalue problems
- ✅ **Optimize performance** by replacing Python loops with vectorized operations
- ✅ **Apply NumPy** as the foundation for Pandas, Matplotlib, Scikit-learn, and Deep Learning

---

## 📖 Study Approach

| Day | Focus Area | Notebook Section |
|-----|------------|------------------|
| **Day 1** | Array Creation, Properties, Reshaping, Basic Math | Cells 1-12 |
| **Day 2** | Statistical Functions, Matrix Multiplication, Visualization | Cells 13-15 |
| **Day 3** | Indexing, Slicing, Boolean Masking, Fancy Indexing, Iteration | Cell 16 |
| **Day 4** | Broadcasting, Vectorization, Ufuncs, Conditional Operations | Cells 17-19 |
| **Day 5** | Stacking, Splitting, Sorting, Searching, Unique Values | Cells 20-26 |
| **Day 6** | Linear Algebra: Transpose, Determinant, Inverse, Eigenvalues | Cells 27-30 |

---

## 🔑 Key Concepts Mastered

### **Performance Optimization**
- **Vectorization**: Element-wise operations on entire arrays (10-100x faster than loops)
- **Broadcasting**: Automatic shape alignment for arithmetic operations
- **Memory Efficiency**: Homogeneous arrays with fixed memory layout
- **Ufuncs**: Optimized C-level implementations for mathematical functions

### **Data Manipulation Patterns**
- **Filtering**: `arr[arr > threshold]` for conditional selection
- **Aggregation**: `np.mean(arr, axis=0)` for row/column statistics
- **Transformation**: `arr.reshape(-1, 1)` for dimension alignment
- **Combination**: `np.vstack/hstack` for building datasets

### **Linear Algebra Foundations**
- Matrix operations essential for ML algorithms
- Eigendecomposition for PCA and dimensionality reduction
- Matrix inversion for solving linear systems
- Rank and trace for matrix property analysis

---

## 📊 Practice Highlights

The notebook includes **practical challenges** such as:
- Student marks analysis (statistics, filtering, sorting)
- Sales data processing (aggregation, ranking)
- Matrix operations with bonus calculations
- Array attribute exploration exercises
- Random data generation and manipulation

---

## 🚀 Status

✅ **NumPy Fundamentals Completed**

---

## 🗺️ Next Steps

| Priority | Topic | Purpose |
|----------|-------|---------|
| **1** | **Pandas** | DataFrames, Series, data cleaning, aggregation, merging |
| **2** | **Matplotlib/Seaborn** | Data visualization, plotting, exploratory analysis |
| **3** | **Scikit-learn** | ML pipelines, preprocessing, model training |
| **4** | **Deep Learning** | TensorFlow/PyTorch foundations |

---

## 💡 Resources for Continued Learning

- [NumPy Official Documentation](https://numpy.org/doc/)
- [NumPy User Guide](https://numpy.org/doc/stable/user/index.html)
- [100 NumPy Exercises](https://github.com/rougier/numpy-100)
- [Python Data Science Handbook - NumPy Chapter](https://jakevdp.github.io/PythonDataScienceHandbook/02.00-introduction-to-numpy.html)

---

*Last Updated: August 2025*  
*Part of the AI Engineer Journey Repository*