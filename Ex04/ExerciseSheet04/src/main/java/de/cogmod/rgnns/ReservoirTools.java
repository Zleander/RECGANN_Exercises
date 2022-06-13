package de.cogmod.rgnns;

import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;

import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.linsol.svd.SolvePseudoInverseSvd_DDRM;

/**
 * @author Sebastian Otte
 */
public class ReservoirTools {
    
    /**
     * Helper function for printing out matrices.
     */
    public static String matrixAsString(final double[][] A) {
        return matrixAsString(A, 2);
    }
    
    /**
     * Helper function for printing out matrices.
     */
    public static String matrixAsString(final double[][] A, final int decimals) {
        //
        final DecimalFormat f = new DecimalFormat();
        f.setDecimalSeparatorAlwaysShown(true);
        f.setMaximumFractionDigits(decimals);
        f.setMinimumFractionDigits(decimals);
        f.setGroupingUsed(false);
        //
        f.setDecimalFormatSymbols(new DecimalFormatSymbols() {
            private static final long serialVersionUID = -2464236658633690492L;
            public char getGroupingSeparator() { return ' '; }
            public char getDecimalSeparator() { return '.'; }
        });
        //
        final StringBuilder out = new StringBuilder();
        //
        final int rows = rows(A);
        final int cols = cols(A);
        //          
        for (int i = 0; i < rows; i++) {
            if (i > 0) out.append("\n");
            //
            // we assume that all rows of A have the same length.
            //
            for (int j = 0; j < cols; j++) {
                if (j > 0) out.append("    ");
                out.append(f.format(A[i][j]));
            }
        }
        //
        return out.toString();
    }
    
    /**
     * Solves the equation system AX = B (A and B are given) via pseudo inverse using
     * a singular value decomposition of B. The result is stored in X.
     */
    public static boolean solveSVD(final double[][] A, final double[][] B, final double[][] X) {
        //
        // convert double array to ejml matrices.
        //
        final DMatrixRMaj AA = new DMatrixRMaj(A);
        //
        final int max = Math.max(AA.numRows, AA.numCols);
        //
        final DMatrixRMaj BB = new DMatrixRMaj(B);
        final DMatrixRMaj XX = new DMatrixRMaj(X);
        //
        // perform least squares...
        //
        final SolvePseudoInverseSvd_DDRM solver = new SolvePseudoInverseSvd_DDRM(max, max);
        if (!solver.setA(AA)) return false;
        //
        solver.solve(BB, XX);
        //
        // write solution back to array.
        //
        for (int i = 0; i < XX.numRows; i++) {
            for (int j = 0; j < XX.numCols; j++) {
                X[i][j] = XX.get(i, j);
            }
        }
        //
        return true;
    }
    

    /**
     * Gives the number of columns of a matrix.
     */
    public static final int cols(final double[][] A) {
        //
        // we assume that all rows of A have the same length.
        //
        return A[0].length;
    }
 
    /**
     * Gives the number of rows of a matrix.
     */
    public static final int rows(final double[][] A) {
        return A.length;
    }
    
    
    /**
     * Computes the matrix multiplication AB. 
     */
    public static double[][] multiply(final double[][] A, final double[][] B) {
        //
        final int rowsA = rows(A);
        final int colsA = cols(A);
        final int rowsB = rows(B);
        final int colsB = cols(B);
        //
        if (colsA != rowsB) {
            throw new RuntimeException("Matrix dimension mismatch.");
        }
        //
        final int rowsC = rowsA;
        final int colsC = colsB;
        //
        final double[][] C = new double[rowsC][colsC];
        //
        multiply(A, B, C);
        //
        return C;
    }

    /**
     * Computes the matrix multiplication AB. 
     */
    public static void multiply(final double[][] A, final double[][] B, final double[][] C) {
        //
        final int rowsA = rows(A);
        final int colsA = cols(A);
        final int rowsB = rows(B);
        final int colsB = cols(B);
        final int rowsC = rows(C);
        final int colsC = cols(C);
        //
        if ((colsA != rowsB) || (rowsC != rowsA) || (colsC != colsB)) {
            throw new RuntimeException("Matrix dimension mismatch.");
        }
        //
        for (int i = 0; i < rowsC; i++) {
            for (int j = 0; j < colsC; j++) {
                //
                double sum = 0.0;
                //
                for (int k = 0; k < colsA; k++) {
                    sum += A[i][k] * B[k][j];
                }
                //
                C[i][j] = sum;
            }
        }
    }
    
    /**
     * Computes the transpose of a matrix.
     */
    public static void transpose(final double[][] A, final double[][] At) {
        //
        final int rows = rows(A);
        final int cols = cols(A);
        //
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                At[j][i] = A[i][j];
            }
        }        
    }

    /**
     * Computes the transpose of a matrix.
     */
    public static double[][] transpose(final double[][] A) {
        final double[][] At = new double[cols(A)][rows(A)];
        transpose(A, At);
        return At;
    }

    /**
     * Helper function for mapping a flat array onto a matrix (row major order).
     */
    public static void map(final double[] values, final int offset, final double[][] A) {
        final int rowsA = rows(A);
        final int colsA = cols(A);
        //
        int idx = offset;
        //
        for (int i = 0; i < rowsA; i++) {
            for (int j = 0; j < colsA; j++) {
                A[i][j] = values[idx];
                idx++;
            }
        }
    }

}

