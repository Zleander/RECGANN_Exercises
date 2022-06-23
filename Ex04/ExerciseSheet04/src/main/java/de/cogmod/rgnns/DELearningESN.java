package de.cogmod.rgnns;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import de.jannlab.io.Serializer;
import de.jannlab.optimization.BasicOptimizationListener;
import de.jannlab.optimization.Objective;
import de.jannlab.optimization.optimizer.DifferentialEvolution;
import de.jannlab.optimization.optimizer.DifferentialEvolution.Mutation;


public class DELearningESN {
	
    public static void main(String[] args) throws Exception {

        double[][] sequence = loadSequence("combat_sequence_5000.txt");

        final int input = 3;
        final int reservoirsize = 20;
        final int output = 3;

        int washout = 100;
        int training = 200;
        int test = 200;

        EchoStateNetwork final_esn = new EchoStateNetwork(input, reservoirsize, output);
        int num_inputweights = final_esn.getInputWeights().length * final_esn.getInputWeights()[0].length;
        System.out.println(num_inputweights);
        //int num_reservoirweights = reservoirsize*(reservoirsize+1);
        //
        // First, we need an objective (fitness) function that
        // we want optimize (minimize). This can be done by implementing
        // the interface Objective.
        //

        double OUTPUT_FEEDBACK_SCALE = 10e-10;

        final Objective f = new Objective() {
            //
            @Override
            public int arity() {
                return final_esn.getWeightsNum();//num_inputweights + num_reservoirweights;
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
                // with the length that is given via arity().

                EchoStateNetwork esn = new EchoStateNetwork(input, reservoirsize, output);
                double[] weights = new double[esn.getWeightsNum()];
                for (int i=0; i < this.arity(); i++){
                    weights[i] = values[offset+i] * (i < num_inputweights ? OUTPUT_FEEDBACK_SCALE : 1);
                }
                esn.writeWeights(weights);
                double error = esn.trainESN(sequence, washout, training, test);

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
        for (int i=0; i < f.arity(); i++){
            solution[i] = solution[i] * (i < num_inputweights ? OUTPUT_FEEDBACK_SCALE : 1);
        }
        // we need to train once again to calculate the output weights (these are not saved in the solution)
        final_esn.writeWeights(solution);
        double error = final_esn.trainESN(sequence, washout, training, test); 
        final_esn.readWeights(solution);
        System.out.println("Final error: "+ error);
        
    	
        String fname = "data/esn-3-" + reservoirsize + "-3.weights";
        Serializer.write(solution, fname);
    }

    
    /**
	 * Helper method for sequence loading from file.
	 */
	public static double[][] loadSequence(final String filename) throws FileNotFoundException, IOException {
        return loadSequence(new FileInputStream(filename));
    }

	/**
	 * Helper method for sequence loading from InputStream.
	 */
    public static double[][] loadSequence(final InputStream inputstream) throws IOException {
        //
        final BufferedReader input = new BufferedReader(
            new InputStreamReader(inputstream));
        //
        final List<String[]> data = new ArrayList<String[]>();
        int maxcols = 0;
        //
        boolean read = true;
        //
        while (read) {
            final String line = input.readLine();
            
            if (line != null) {
                final String[] components = line.trim().split("\\s*(,|\\s)\\s*");
                final int cols = components.length;
                if (cols > maxcols) {
                    maxcols = cols;
                }
                data.add(components);
            } else {
                read = false;
            }
        }
        input.close();
        //
        final int cols = maxcols;
        final int rows = data.size();
        //
        if ((cols == 0) || (rows == 0)) return null;
        //
        final double[][] result = new double[rows][cols];
        //
        for (int r = 0; r < rows; r++) {
            String[] elements = data.get(r);
            for (int c = 0; c < cols; c++) {
                final double value = Double.parseDouble(elements[c]);
                result[r][c] = value;
            }
        }
        //
        return result;
    }
}