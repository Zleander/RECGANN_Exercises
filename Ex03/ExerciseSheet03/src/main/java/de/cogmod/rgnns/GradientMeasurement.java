package de.cogmod.rgnns;

import java.util.Random;

import de.cogmod.rgnns.misc.BasicLearningListener;
import de.cogmod.rgnns.misc.Spiral;
import de.cogmod.rgnns.misc.TrajectoryGenerator;

public class GradientMeasurement {
	
	private static Random rnd = new Random(100L);

	public GradientMeasurement() {
		//rnn = new RecurrentNeuralNetwork();
	}

	public static void main(String[] args) {
		//
        final int trainlength         = 100;
        //
        final double[][] input  = new double[trainlength][1];
        final double[][] target = new double[1][1];
        //
        // set all values of the input sequence to 1.0.
        //
        for (int i=0; i<trainlength; i++) {
        	input[i][0] = 1.0;
        }
        target[0][0] = 1;
        //
        final RecurrentNeuralNetwork net = new RecurrentNeuralNetwork(1, 1, 1);
        //
        // we disable all biases.
        //
        net.setBias(1, false);
        net.setBias(2, false);
        //
        // perform training.
        //
        final int epochs = 100000; // 100000
        final double learningrate =  0.00002; // 0.00002
        final double momentumrate = 0.95; // 0.95
        //
        // generate initial weights and prepare the RNN buffer
        // for BPTT over the required number of time steps.
        //
        net.initializeWeights(rnd, 0.1);
        net.rebufferOnDemand(trainlength);
        
        //perform forward and backwards pass
        net.forwardPass(input);
        net.backwardPass(target);
        double[][][] delta = net.getDelta();
        for (int t=trainlength-1; t>=0; t--) {
        	System.out.println(delta[1][0][t]);
        }
   

	}

}
