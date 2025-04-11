import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.Callable;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

public class StrassenMultithreaded {

    private static final int THRESHOLD = 64;
    private static int numThreads;
    private static ExecutorService executor;

    public static void main(String[] args) {
        int n = 2000; // Matrix size, must be a power of 2 for simplicity
        numThreads = 6;

        executor = Executors.newFixedThreadPool(numThreads);

        System.out.println("Matrix size: " + n + "x" + n);
        System.out.println("Number of threads: " + numThreads);

        int[][] a = initializeMatrix(n);
        int[][] b = initializeMatrix(n);

        long startTime = System.currentTimeMillis();
        int[][] result = strassen(n, a, b);
        long endTime = System.currentTimeMillis();

        long duration = endTime - startTime;
        System.out.println("Execution time: " + duration + " ms");

        if (n <= 100) {
            System.out.println("Verifying result...");
            if (verifyResult(n, a, b, result)) {
                System.out.println("Result is correct!");
            } else {
                System.out.println("Result is incorrect!");
            }
        }

        if (n <= 10) {
            System.out.println("\nMatrix A:");
            printMatrix(a);

            System.out.println("\nMatrix B:");
            printMatrix(b);

            System.out.println("\nResult Matrix:");
            printMatrix(result);
        }

        executor.shutdown();
        try {
            if (!executor.awaitTermination(10, TimeUnit.SECONDS)) {
                executor.shutdownNow();
            }
        } catch (InterruptedException e) {
            executor.shutdownNow();
        }
    }

    private static int[][] initializeMatrix(int n) {
        int[][] matrix = new int[n][n];
        Random rand = new Random();

        List<Future<?>> futures = new ArrayList<>();
        int chunkSize = Math.max(1, n / numThreads);

        for (int t = 0; t < numThreads; t++) {
            final int startRow = t * chunkSize;
            final int endRow = (t == numThreads - 1) ? n : (t + 1) * chunkSize;

            futures.add(executor.submit(() -> {
                for (int i = startRow; i < endRow; i++) {
                    for (int j = 0; j < n; j++) {
                        matrix[i][j] = rand.nextInt(10);
                    }
                }
                return null;
            }));
        }

        for (Future<?> future : futures) {
            try {
                future.get();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        return matrix;
    }

    private static void printMatrix(int[][] matrix) {
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                System.out.printf("%5d ", matrix[i][j]);
            }
            System.out.println();
        }
    }

    private static int[][] addMatrices(int n, int[][] mat1, int[][] mat2, boolean add) {
        int[][] result = new int[n][n];

        List<Future<?>> futures = new ArrayList<>();
        int chunkSize = Math.max(1, n / numThreads);

        for (int t = 0; t < numThreads; t++) {
            final int startRow = t * chunkSize;
            final int endRow = (t == numThreads - 1) ? n : (t + 1) * chunkSize;

            futures.add(executor.submit(() -> {
                for (int i = startRow; i < endRow; i++) {
                    for (int j = 0; j < n; j++) {
                        if (add) {
                            result[i][j] = mat1[i][j] + mat2[i][j];
                        } else {
                            result[i][j] = mat1[i][j] - mat2[i][j];
                        }
                    }
                }
                return null;
            }));
        }

        for (Future<?> future : futures) {
            try {
                future.get();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        return result;
    }

    private static int[][] standardMultiply(int n, int[][] a, int[][] b) {
        int[][] result = new int[n][n];

        List<Future<?>> futures = new ArrayList<>();
        int chunkSize = Math.max(1, n / numThreads);

        for (int t = 0; t < numThreads; t++) {
            final int startRow = t * chunkSize;
            final int endRow = (t == numThreads - 1) ? n : (t + 1) * chunkSize;

            futures.add(executor.submit(() -> {
                for (int i = startRow; i < endRow; i++) {
                    for (int j = 0; j < n; j++) {
                        result[i][j] = 0;
                        for (int k = 0; k < n; k++) {
                            result[i][j] += a[i][k] * b[k][j];
                        }
                    }
                }
                return null;
            }));
        }

        for (Future<?> future : futures) {
            try {
                future.get();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        return result;
    }

    private static int[][] strassen(int n, int[][] a, int[][] b) {
        if (n <= THRESHOLD) {
            return standardMultiply(n, a, b);
        }

        if (n == 1) {
            int[][] result = new int[1][1];
            result[0][0] = a[0][0] * b[0][0];
            return result;
        }

        int m = n / 2;

        int[][] a11 = new int[m][m];
        int[][] a12 = new int[m][m];
        int[][] a21 = new int[m][m];
        int[][] a22 = new int[m][m];
        int[][] b11 = new int[m][m];
        int[][] b12 = new int[m][m];
        int[][] b21 = new int[m][m];
        int[][] b22 = new int[m][m];

        List<Future<?>> splitFutures = new ArrayList<>();

        splitFutures.add(executor.submit(() -> {
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    a11[i][j] = a[i][j];
                    b11[i][j] = b[i][j];
                }
            }
        }));

        splitFutures.add(executor.submit(() -> {
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    a12[i][j] = a[i][j + m];
                    b12[i][j] = b[i][j + m];
                }
            }
        }));

        splitFutures.add(executor.submit(() -> {
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    a21[i][j] = a[i + m][j];
                    b21[i][j] = b[i + m][j];
                }
            }
        }));

        splitFutures.add(executor.submit(() -> {
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    a22[i][j] = a[i + m][j + m];
                    b22[i][j] = b[i + m][j + m];
                }
            }
        }));

        for (Future<?> future : splitFutures) {
            try {
                future.get();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        int[][] s1 = addMatrices(m, b12, b22, false);
        int[][] s2 = addMatrices(m, a11, a12, true);
        int[][] s3 = addMatrices(m, a21, a22, true);
        int[][] s4 = addMatrices(m, b21, b11, false);
        int[][] s5 = addMatrices(m, a11, a22, true);
        int[][] s6 = addMatrices(m, b11, b22, true);
        int[][] s7 = addMatrices(m, a12, a22, false);
        int[][] s8 = addMatrices(m, b21, b22, true);
        int[][] s9 = addMatrices(m, a11, a21, false);
        int[][] s10 = addMatrices(m, b11, b12, true);

        int[][] p1 = strassen(m, a11, s1);
        int[][] p2 = strassen(m, s2, b22);
        int[][] p3 = strassen(m, s3, b11);
        int[][] p4 = strassen(m, a22, s4);
        int[][] p5 = strassen(m, s5, s6);
        int[][] p6 = strassen(m, s7, s8);
        int[][] p7 = strassen(m, s9, s10);

        int[][] c11 = addMatrices(m, addMatrices(m, p5, p4, true), addMatrices(m, p2, p6, false), true);
        int[][] c12 = addMatrices(m, p1, p2, true);
        int[][] c21 = addMatrices(m, p3, p4, true);
        int[][] c22 = addMatrices(m, addMatrices(m, p5, p1, true), addMatrices(m, p3, p7, true), false);

        int[][] result = new int[n][n];

        List<Future<?>> combineFutures = new ArrayList<>();

        combineFutures.add(executor.submit(() -> {
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    result[i][j] = c11[i][j];
                    result[i][j + m] = c12[i][j];
                    result[i + m][j] = c21[i][j];
                    result[i + m][j + m] = c22[i][j];
                }
            }
        }));

        for (Future<?> future : combineFutures) {
            try {
                future.get();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        return result;
    }

    private static int nextPowerOfTwo(int n) {
        int power = 1;
        while (power < n) {
            power *= 2;
        }
        return power;
    }

    private static int[][] padMatrix(int oldSize, int newSize, int[][] matrix) {
        int[][] paddedMatrix = new int[newSize][newSize];
        for (int i = 0; i < oldSize; i++) {
            System.arraycopy(matrix[i], 0, paddedMatrix[i], 0, oldSize);
        }
        return paddedMatrix;
    }

    private static int[][] extractResult(int oldSize, int newSize, int[][] paddedResult) {
        int[][] result = new int[oldSize][oldSize];
        for (int i = 0; i < oldSize; i++) {
            System.arraycopy(paddedResult[i], 0, result[i], 0, oldSize);
        }
        return result;
    }

    private static boolean verifyResult(int n, int[][] a, int[][] b, int[][] result) {
        int[][] expected = new int[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    expected[i][j] += a[i][k] * b[k][j];
                }
            }
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (expected[i][j] != result[i][j]) {
                    return false;
                }
            }
        }
        return true;
    }

    static class ResultQuadrant {
        int index;
        int[][] matrix;

        public ResultQuadrant(int index, int[][] matrix) {
            this.index = index;
            this.matrix = matrix;
        }
    }
}
