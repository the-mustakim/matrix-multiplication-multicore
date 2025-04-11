#include <iostream>
#include <omp.h>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <vector>
#include <iomanip>
#include <cstring>

using namespace std;

#define THRESHOLD 64

// Function to allocate memory for a matrix
int** allocateMatrix(int n) {
    int** mat = new int*[n];
    for (int i = 0; i < n; i++) {
        mat[i] = new int[n];
    }
    return mat;
}

// Function to free the allocated memory for a matrix
void freeMatrix(int n, int** mat) {
    for (int i = 0; i < n; i++) {
        delete[] mat[i];
    }
    delete[] mat;
}

// Function to initialize a matrix with random values
void initializeMatrix(int n, int** mat) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            mat[i][j] = rand() % 10;
        }
    }
}

// Function to print a matrix
void printMatrix(int n, int** mat) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << setw(5) << mat[i][j] << " ";
        }
        cout << endl;
    }
}

// Function to add or subtract two matrices
int** addMatrices(int n, int** mat1, int** mat2, bool add) {
    int** result = allocateMatrix(n);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (add) {
                result[i][j] = mat1[i][j] + mat2[i][j];
            } else {
                result[i][j] = mat1[i][j] - mat2[i][j];
            }
        }
    }
    return result;
}

// Function for standard matrix multiplication (for small matrices)
int** standardMultiply(int n, int** a, int** b) {
    int** result = allocateMatrix(n);
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            result[i][j] = 0;
            for (int k = 0; k < n; k++) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    
    return result;
}

// Function to multiply two matrices using Strassen's algorithm
int** strassen(int n, int** a, int** b) {
    // For small matrices, use standard multiplication
    if (n <= THRESHOLD) {
        return standardMultiply(n, a, b);
    }
    
    // For matrices of size 1, simple multiplication
    if (n == 1) {
        int** result = allocateMatrix(1);
        result[0][0] = a[0][0] * b[0][0];
        return result;
    }

    int m = n / 2;
    
    // Divide matrices a and b into 4 submatrices
    int** a11 = allocateMatrix(m);
    int** a12 = allocateMatrix(m);
    int** a21 = allocateMatrix(m);
    int** a22 = allocateMatrix(m);
    int** b11 = allocateMatrix(m);
    int** b12 = allocateMatrix(m);
    int** b21 = allocateMatrix(m);
    int** b22 = allocateMatrix(m);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            a11[i][j] = a[i][j];
            a12[i][j] = a[i][j + m];
            a21[i][j] = a[i + m][j];
            a22[i][j] = a[i + m][j + m];
            b11[i][j] = b[i][j];
            b12[i][j] = b[i][j + m];
            b21[i][j] = b[i + m][j];
            b22[i][j] = b[i + m][j + m];
        }
    }

    // Calculate intermediate matrices using Strassen's formulas
    int** s1, **s2, **s3, **s4, **s5, **s6, **s7, **s8, **s9, **s10;
    
    #pragma omp parallel sections
    {
        #pragma omp section
        s1 = addMatrices(m, b12, b22, false);  // s1 = b12 - b22
        
        #pragma omp section
        s2 = addMatrices(m, a11, a12, true);   // s2 = a11 + a12
        
        #pragma omp section
        s3 = addMatrices(m, a21, a22, true);   // s3 = a21 + a22
        
        #pragma omp section
        s4 = addMatrices(m, b21, b11, false);  // s4 = b21 - b11
        
        #pragma omp section
        s5 = addMatrices(m, a11, a22, true);   // s5 = a11 + a22
        
        #pragma omp section
        s6 = addMatrices(m, b11, b22, true);   // s6 = b11 + b22
        
        #pragma omp section
        s7 = addMatrices(m, a12, a22, false);  // s7 = a12 - a22
        
        #pragma omp section
        s8 = addMatrices(m, b21, b22, true);   // s8 = b21 + b22
        
        #pragma omp section
        s9 = addMatrices(m, a11, a21, false);  // s9 = a11 - a21
        
        #pragma omp section
        s10 = addMatrices(m, b11, b12, true);  // s10 = b11 + b12
    }

    // Recursive calls for strassen multiplication
    int** p1, **p2, **p3, **p4, **p5, **p6, **p7;
    
    #pragma omp parallel sections
    {
        #pragma omp section
        p1 = strassen(m, a11, s1);            // p1 = a11 * (b12 - b22)
        
        #pragma omp section
        p2 = strassen(m, s2, b22);            // p2 = (a11 + a12) * b22
        
        #pragma omp section
        p3 = strassen(m, s3, b11);            // p3 = (a21 + a22) * b11
        
        #pragma omp section
        p4 = strassen(m, a22, s4);            // p4 = a22 * (b21 - b11)
        
        #pragma omp section
        p5 = strassen(m, s5, s6);             // p5 = (a11 + a22) * (b11 + b22)
        
        #pragma omp section
        p6 = strassen(m, s7, s8);             // p6 = (a12 - a22) * (b21 + b22)
        
        #pragma omp section
        p7 = strassen(m, s9, s10);            // p7 = (a11 - a21) * (b11 + b12)
    }

    // Free intermediate matrices that are no longer needed
    freeMatrix(m, s1);
    freeMatrix(m, s2);
    freeMatrix(m, s3);
    freeMatrix(m, s4);
    freeMatrix(m, s5);
    freeMatrix(m, s6);
    freeMatrix(m, s7);
    freeMatrix(m, s8);
    freeMatrix(m, s9);
    freeMatrix(m, s10);

    // Calculate the four quadrants of the result matrix
    int** temp1, **temp2, **temp3, **temp4, **c11, **c12, **c21, **c22;
    
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            temp1 = addMatrices(m, p5, p4, true);   // temp1 = p5 + p4
            temp2 = addMatrices(m, temp1, p2, false); // temp2 = temp1 - p2
            c11 = addMatrices(m, temp2, p6, true);  // c11 = temp2 + p6
        }
        
        #pragma omp section
        {
            c12 = addMatrices(m, p1, p2, true);     // c12 = p1 + p2
        }
        
        #pragma omp section
        {
            c21 = addMatrices(m, p3, p4, true);     // c21 = p3 + p4
        }
        
        #pragma omp section
        {
            temp3 = addMatrices(m, p5, p1, true);   // temp3 = p5 + p1
            temp4 = addMatrices(m, temp3, p3, false); // temp4 = temp3 - p3
            c22 = addMatrices(m, temp4, p7, false); // c22 = temp4 - p7
        }
    }

    // Free intermediate matrices
    freeMatrix(m, p1);
    freeMatrix(m, p2);
    freeMatrix(m, p3);
    freeMatrix(m, p4);
    freeMatrix(m, p5);
    freeMatrix(m, p6);
    freeMatrix(m, p7);
    
    #pragma omp parallel sections
    {
        #pragma omp section
        if (temp1) freeMatrix(m, temp1);
        
        #pragma omp section
        if (temp2) freeMatrix(m, temp2);
        
        #pragma omp section
        if (temp3) freeMatrix(m, temp3);
        
        #pragma omp section
        if (temp4) freeMatrix(m, temp4);
    }

    // Allocate the final matrix and combine the four submatrices
    int** result = allocateMatrix(n);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            result[i][j] = c11[i][j];
            result[i][j + m] = c12[i][j];
            result[i + m][j] = c21[i][j];
            result[i + m][j + m] = c22[i][j];
        }
    }

    // Free the submatrices
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            freeMatrix(m, a11);
            freeMatrix(m, a12);
        }
        
        #pragma omp section
        {
            freeMatrix(m, a21);
            freeMatrix(m, a22);
        }
        
        #pragma omp section
        {
            freeMatrix(m, b11);
            freeMatrix(m, b12);
        }
        
        #pragma omp section
        {
            freeMatrix(m, b21);
            freeMatrix(m, b22);
        }
        
        #pragma omp section
        {
            freeMatrix(m, c11);
            freeMatrix(m, c12);
        }
        
        #pragma omp section
        {
            freeMatrix(m, c21);
            freeMatrix(m, c22);
        }
    }

    return result;
}

// Function to ensure matrix size is a power of 2
int nextPowerOfTwo(int n) {
    int power = 1;
    while (power < n) {
        power *= 2;
    }
    return power;
}

// Function to pad a matrix to the next power of 2
int** padMatrix(int original_size, int new_size, int** original) {
    int** padded = allocateMatrix(new_size);
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < new_size; i++) {
        for (int j = 0; j < new_size; j++) {
            if (i < original_size && j < original_size) {
                padded[i][j] = original[i][j];
            } else {
                padded[i][j] = 0;
            }
        }
    }
    
    return padded;
}

// Function to extract the result from the padded matrix
int** extractResult(int original_size, int padded_size, int** padded) {
    int** result = allocateMatrix(original_size);
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < original_size; i++) {
        for (int j = 0; j < original_size; j++) {
            result[i][j] = padded[i][j];
        }
    }
    
    return result;
}

// Test function to verify the correctness of matrix multiplication
bool verifyResult(int n, int** a, int** b, int** result) {
    int** standard = standardMultiply(n, a, b);
    bool correct = true;
    
    for (int i = 0; i < n && correct; i++) {
        for (int j = 0; j < n && correct; j++) {
            if (standard[i][j] != result[i][j]) {
                correct = false;
            }
        }
    }
    
    freeMatrix(n, standard);
    return correct;
}

int main(int argc, char* argv[]) {
    // Seed random number generator
    srand(time(NULL));
    
    // Default values
    int n = 2000; // Matrix size
    int num_threads = 12;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            num_threads = atoi(argv[i + 1]);
            omp_set_num_threads(num_threads);
            i++;
        }
    }
    
    cout << "Matrix size: " << n << "x" << n << endl;
    cout << "Number of threads: " << num_threads << endl;
    
    // Initialize matrices a and b
    int** a = allocateMatrix(n);
    int** b = allocateMatrix(n);
    initializeMatrix(n, a);
    initializeMatrix(n, b);
    
    // Calculate next power of 2 for matrix size
    int padded_size = nextPowerOfTwo(n);
    int** padded_a = a;
    int** padded_b = b;
    
    // Pad matrices if necessary
    if (padded_size != n) {
        cout << "Padding matrices to " << padded_size << "x" << padded_size << endl;
        padded_a = padMatrix(n, padded_size, a);
        padded_b = padMatrix(n, padded_size, b);
    }
    
    // Perform matrix multiplication using Strassen's algorithm
    auto start_time = chrono::high_resolution_clock::now();
    int** padded_result = strassen(padded_size, padded_a, padded_b);
    auto end_time = chrono::high_resolution_clock::now();
    
    // Extract the result if padding was used
    int** result = padded_result;
    if (padded_size != n) {
        result = extractResult(n, padded_size, padded_result);
        freeMatrix(padded_size, padded_result);
    }
    
    // Calculate and print execution time
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
    cout << "Execution time: " << duration << " ms" << endl;
    
    // Verify the result for small matrices (optional)
    if (n <= 100) {
        cout << "Verifying result..." << endl;
        if (verifyResult(n, a, b, result)) {
            cout << "Result is correct!" << endl;
        } else {
            cout << "Result is incorrect!" << endl;
        }
    }
    
    // Print the result for very small matrices (optional)
    if (n <= 10) {
        cout << "\nMatrix A:" << endl;
        printMatrix(n, a);
        
        cout << "\nMatrix B:" << endl;
        printMatrix(n, b);
        
        cout << "\nResult Matrix:" << endl;
        printMatrix(n, result);
    }
    
    // Free allocated memory
    freeMatrix(n, a);
    freeMatrix(n, b);
    freeMatrix(n, result);
    
    // Free padded matrices if they were created
    if (padded_size != n) {
        freeMatrix(padded_size, padded_a);
        freeMatrix(padded_size, padded_b);
    }
    
    return 0;
}