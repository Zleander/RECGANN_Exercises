package de.cogmod.rgnns;

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
        int reservoirsize = reservoirweights.length;
        int outputsize = this.getAct()[this.getOutputLayer()].length;

        // we only have to set the output weights, using least squates optimization, so W_out = pseudoinv(X) @ Z,
        // where X is the matrix of hidden activations of size (training, reservoirsize+1),
        // and Z is the matrix of size (training, outputsize) that contains the targets (in our case, the sequence values at the next timestep).
        // W_out has size (reservoirsize+1, outputsize). (+1 because of biases)

        // First we need to calculate the matrix X by repeatedly executing forward passes through the ESN.
        // Execute washout and training phase using teacher forcing:
        double[][] X = new double[training][reservoirsize+1];
        forwardPass(sequence[0]);
        for (int t = 1; t < washout+training; t++){
            teacherForcing(sequence[t]);
            forwardPassOscillator();
            if (t >= washout) {
                for (int j = 0; j < reservoirsize; j++) { // X[t,:] = hidden activations
                    final double[][][] act = this.getAct();
                    X[t-washout][j] = act[1][j][0]; // 1 is the hidden layer, j is the neuron and t is the timestep
                }
                X[t-washout][reservoirsize] = 1;
            }
        }

        // extract X as the activations from the training phase
        
        /*for (int t = washout; t < washout+training; t++){
            for (int j = 0; j < reservoirsize; j++) { // X[t,:] = hidden activations
                final double[][][] act = this.getAct();
                X[t][j] = act[1][j][t]; // 1 is the hidden layer, j is the neuron and t is the timestep
            }
        }*/

        // construct Z = sequence[washout+1:training+washout+1,:]
        double[][] Z = new double[training][outputsize];
        for (int t = washout+1; t < washout+training+1; t++){
            for (int j = 0; j < outputsize; j++) {
                Z[t-washout-1][j] = sequence[t][j];
            }
        }

        // solve the linear equation system and store weights
        ReservoirTools.solveSVD(X, Z, this.outputweights);

        // Now evaluate the network in the test phase.
        double[][] predictions = new double[test][outputsize];
        double[][] test_targets = new double[test][outputsize];
        for (int t = 0; t < test; t++) {
            forwardPassOscillator();
            for (int i = 0; i < outputsize; i++) {
                predictions[t][i] = this.getAct()[this.getOutputLayer()][i][0]; // oh we could probably use the return value from the function above
                test_targets[t][i] = sequence[t+washout+training][i];
            }
        }
        double error = RMSE(predictions, test_targets);
        
        return error; // error.
    }
    
}