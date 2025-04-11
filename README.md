# 🧮 Multicore Matrix Multiplication - Java & OpenMP

This project demonstrates efficient multiplication of large dense matrices using two approaches:

1. ✅ **Multithreaded Java Program** using `ExecutorService`  
2. ⚙️ **OpenMP-based CPP Program** for parallel execution on multiple cores
3. 
---

## 📁 Directory Structure

```
.
├── StrassenMultithreaded.java       # Java implementation
├── StrassenOMP.c                    # C/OpenMP implementation
├── report.docx                      # Design, analysis, and results
└── README.md                        # Project documentation
```

---

## 🚀 Java Program (Multithreaded)

### 💻 Compile & Run
```bash
javac StrassenMultithreaded.java
java StrassenMultithreaded
```

### ✅ Features
- Uses **ExecutorService** to manage thread pools.
- Dynamically partitions matrix blocks.
- Blocked algorithm to enhance cache performance.

---

## ⚙️ OpenMP Program (C)

### 💻 Compile & Run
```bash
gcc -fopenmp StrassenOMP.c -o StrassenOMP
./StrassenOMP
```

### ✅ Features
- Uses `#pragma omp parallel for` for parallel loops.
- Schedule tuning (`static`, `dynamic`) for optimal performance.
- Efficient use of multiple cores for large matrices.

---

## 🧠 Design Overview

- **Blocked Matrix Multiplication** used in both implementations.
- Improves spatial and temporal locality for better cache usage.
- Algorithms are designed to distribute workload evenly among available CPU cores.

---

## 📊 Testing & Performance

### ✅ Test Parameters
- **Matrix Sizes:** 500x500, 1000x1000, 2000x2000  
- **Core Counts:** 2, 4, 8

### 📈 Metrics Collected
- Execution Time
- Speedup
- Efficiency
- Cache performance (qualitative)

### 📄 Refer to `report.docx` for:
- Algorithm design explanations
- Observations and discussion

---

## 📌 Notes

- For reproducible results, run on a multi-core CPU without heavy background processes.
- GCC with OpenMP support must be installed to run the CPP program.
- Java program uses all available cores automatically.
  
