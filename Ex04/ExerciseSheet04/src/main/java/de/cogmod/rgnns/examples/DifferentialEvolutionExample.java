package de.cogmod.rgnns.examples;

import de.jannlab.optimization.BasicOptimizationListener;
import de.jannlab.optimization.Objective;
import de.jannlab.optimization.optimizer.DifferentialEvolution;
import de.jannlab.optimization.optimizer.DifferentialEvolution.Mutation;

import static de.cogmod.rgnns.ReservoirTools.*;

import java.util.Random;

/**
 * @author Sebastian Otte
 */
public class DifferentialEvolutionExample {
   
    /**
     * Small helper function for x * x. 
     */
    public static double sq(final double x) {
        return x * x;
    }
    
    public static void main(String[] args) {
        //
        // In this example, we use DifferentialEvolution
        // so solve the same least squares problem as in
        // ReservoirToolsExample.
        //
        final double[][] A = ReservoirToolsExample.A;
        final double[][] b = ReservoirToolsExample.b;
        //
        final int rowsb = rows(b);
        final int colsb = cols(b);
        //
        final int rowsx = cols(A);
        final int colsx = cols(b);
        final double[][] x = new double[rowsx][colsx];
        final int sizex = rowsx * colsx;
        //
        // First, we need an objective (fitness) function that
        // we want optimize (minimize). This can be done by implementing
        // the interface Objective.
        //
        final Objective f = new Objective() {
            //
            @Override
            public int arity() {
                return sizex;
            }
            @Override
            /**
             * This is the callback method that is called from the 
             * optimizer to compute the "fitness" of a particular individual.
             */
            public double compute(double[] values, int offset) {
                //
                // the parameters for which the optimizer requests a fitness
                // value or stored in values starting at the given offset
                // with the length that is given via arity(), namely, sizex.
                //
                final double[][] x = new double[rowsx][colsx];
                map(values, offset, x);
                //
                // Compute A * x.
                //
                final double[][] Ax = multiply(A, x);
                //
                // compute square error of x and b.
                //
                double error = 0.0;
                //
                for (int i = 0; i < rowsb; i++) {
                    for (int j = 0; j < colsb; j++) {
                        error += sq(b[i][j] - Ax[i][j]);
                    }
                }
                return error;
            }
        };
        //
        // Now we setup the optimizer.
        //
        final DifferentialEvolution optimizer = new DifferentialEvolution();
        //
        // The same parameters can be used for reservoir optimization.
        //
        optimizer.setF(0.4);
        optimizer.setCR(0.6);
        optimizer.setPopulationSize(5);
        optimizer.setMutation(Mutation.CURR2RANDBEST_ONE);
        //
        optimizer.setInitLbd(-0.1);
        optimizer.setInitUbd(0.1);
        //
        // Obligatory things...
        // 
        optimizer.setRnd(new Random(1234));
        optimizer.setParameters(f.arity());
        optimizer.updateObjective(f);
        //
        // for observing the optimization process.
        //
        optimizer.addListener(new BasicOptimizationListener());
        //
        optimizer.initialize();
        //
        // go!
        //
        optimizer.iterate(1000, 0.0);
        //
        // read the best solution.
        //
        final double[] solution = new double[f.arity()];
        optimizer.readBestSolution(solution, 0);
        map(solution, 0, x);
        //
        // Print out solution. Note that least squares solution is:
        // 0.68
        // -0.07
        // -0.10
        // -0.05
        // 0.69
        //
        System.out.println();
        System.out.println("Evolved solution for Ax = b");
        System.out.println(matrixAsString(x, 2));
        
    }
 
    
}