package de.cogmod.rgnns.examples;

import static de.cogmod.rgnns.ReservoirTools.*;

/**
 * @author Sebastian Otte
 */
public class ReservoirToolsExample {

    public final static double[][] A = {
        { 2,  6,  4,  6, 7},
        { 3, -4,  2,  3, 3},
        { 3,  1,  1, -2, 0},
        {-2,  5,  3,  4, 1},
        { 3,  1,  1,  2, 0},
        { 6, -1,  4, -2, 5},
        { 1, -1, -3,  2, 1},
        { 1, -3,  2, -1, 2},
        { 3,  2,  3, -1, 6},
    };
    //
    public final static double[][] b = {
        {1},
        {2},
        {3},
        {4},
        {5},
        {6},
        {7},
        {8},
        {9}
    };
    
    public static void main(String[] args) {
        //
        final double[][] x = new double[cols(A)][cols(b)];
        //
        System.out.println("A");
        System.out.println(matrixAsString(A));
        //
        System.out.println();
        System.out.println("b");
        System.out.println(matrixAsString(b));
        //
        solveSVD(A, b, x);
        //
        System.out.println();
        System.out.println("Solution x of equation Ax = b");
        System.out.println(matrixAsString(x, 2));
        //
        System.out.println();
        System.out.println();
        System.out.println("transpose(A)");
        System.out.println(matrixAsString(transpose(A), 2));
        //
        System.out.println();
        System.out.println();
        System.out.println("A * At");
        System.out.println(matrixAsString(
            multiply(A, transpose(A))
        ));
        //
        System.out.println();
        System.out.println();
        System.out.println("At * A");
        System.out.println(matrixAsString(
            multiply(transpose(A), A)
        ));    
    }  
}