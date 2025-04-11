# Multicore Matrix Multiplication - Java & OpenMP

This project implements matrix multiplication using:
1. Multithreaded Java (ExecutorService)
2. Parallel CPP using OpenMP

## 🚀 Java Program

### Compile & Run:
```bash
javac StrassenMultithreaded.java
java StrassenMultithreaded

## ⚙️ OpenMP Program

###Compile & Run:
```bash
gcc -fopenmp StrassenOMP.c -o StrassenOMP
./StrassenOMP

## 🧠 Design
Blocked Matrix Multiplication improves cache efficiency.
Workload is evenly distributed among cores.
Java: Uses ExecutorService and dynamic partitioning of matrix blocks.
OpenMP: Uses #pragma omp parallel for with schedule tuning.

##📊 Testing & Performance
Tested on:
Matrix Sizes: 500x500, 1000x1000, 2000x2000
Cores: 2, 4, 8
Metrics recorded:
Execution Time
Speedup Anaslysis
Analysis and Discussion
Please refer to report.docx for full details.
