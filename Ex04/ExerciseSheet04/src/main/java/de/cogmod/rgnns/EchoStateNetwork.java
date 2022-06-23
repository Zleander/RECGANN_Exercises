package de.cogmod.rgnns;

import java.util.Arrays;

/**
 * @author Sebastian Otte
 */
public class EchoStateNetwork extends RecurrentNeuralNetwork {

    private double[][] inputweights;
    private double[][] reservoirweights;
    private double[][] outputweights;
    
    public double[][] getInputWeights() {
        return this.inputweights;
    }
    
    public double[][] getReservoirWeights() {
        return this.reservoirweights;
    }
    
    public double[][] getOutputWeights() {
        return this.outputweights;
    }
    
    public EchoStateNetwork(
        final int input,
        final int reservoirsize,
        final int output
    ) {
        super(input, reservoirsize, output);
        //
        this.inputweights     = this.getWeights()[0][1];
        this.reservoirweights = this.getWeights()[1][1];
        this.outputweights    = this.getWeights()[1][2];
        //
    }
    
    @Override
    public void rebufferOnDemand(int sequencelength) {
        super.rebufferOnDemand(1);
    }
    

    public double[] output() {
        //
        final double[][][] act = this.getAct();
        final int outputlayer  = this.getOutputLayer(); 
        //
        final int n = act[outputlayer].length;
        //
        final double[] result = new double[n];
        final int t = Math.max(0, this.getLastInputLength() - 1);
        //
        for (int i = 0; i < n; i++) {
            result[i] = act[outputlayer][i][t];
        }
        //
        return result;
    }
    
    /**
     * This is an ESN specific forward pass realizing 
     * an oscillator by means of an output feedback via
     * the input layer. This method requires that the input
     * layer size matches the output layer size. 
     */
    public double[] forwardPassOscillator() {
        //
        // this method causes an additional copy operation
        // but it is more readable from outside.
        //
        final double[] output = this.output();
        return this.forwardPass(output);
    }
    
    /**
     * Overwrites the current output with the given target.
     */
    public void teacherForcing(final double[] target) {
        //
        final double[][][] act = this.getAct();
        final int outputlayer  = this.getOutputLayer(); 
        //
        final int n = act[outputlayer].length;
        //
        final int t = this.getLastInputLength() - 1;
        //
        for (int i = 0; i < n; i++) {
            act[outputlayer][i][t] = target[i];
        }
    }
    
    /**
     * ESN training algorithm. 
     */
    public double trainESN(
        final double[][] sequence,
        final int washout,
        final int training,
        final int test
    ) {
        //split the sequence into parts for training and testing
        double[][] train_seq = Arrays.copyOfRange(sequence, washout, washout+training);
        double[][] test_seq = Arrays.copyOfRange(sequence, washout+training, washout+training+test);

        for (int t=0; t<washout; t++) {
            forwardPassOscillator();
            teacherForcing(sequence[t]);
        }
        // extract the hidden activations X
        double[][] X = new double[training][reservoirweights.length+1];
        for (int t=0; t<training; t++) {
            forwardPassOscillator();
            final double[][][] act = this.getAct();
            for (int j = 0; j < reservoirweights.length; j++) { // X[t,:] = hidden activations
                X[t][j] = act[1][j][0]; // 1 is the hidden layer, j is the neuron and t is the timestep
            }
            X[t][reservoirweights.length] = 1;
            teacherForcing(sequence[washout+t]);
        }
        //solve the least squares problem
        ReservoirTools.solveSVD(X, train_seq, this.getWeights()[1][2]);

        double[][] X_test = new double[test][this.getLastInputLength()];
        for (int t=0; t<test; t++) {
            X_test[t] = forwardPassOscillator();
        }

        return RMSE(X_test, test_seq);
    }
    
}