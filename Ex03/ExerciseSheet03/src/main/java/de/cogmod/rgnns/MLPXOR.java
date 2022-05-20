package de.cogmod.rgnns;

import java.util.Random;

import de.cogmod.rgnns.misc.BasicLearningListener;


/**
 * @author Sebastian Otte
 */
public class MLPXOR {
    
    public static void main(String[] args) {
        //
        final double[][] input = {
            {0, 0}, {0, 1}, {1, 0}, {1, 1}
        };
        final double[][] target = {
            {0}, {1}, {1}, {0}
        };
        //
        //final Random rnd = new Random(100);
        final Random rnd = new Random(System.currentTimeMillis());
        //
        // set up network. biases are used by default, but
        // be deactivated using net.setBias(layer, false),
        // where layer gives the layer index (1 = is the first hidden layer).
        //
        final MultiLayerPerceptron net = new MultiLayerPerceptron(2, 2, 1);
        //
        // biases can be deactivated using
        // net.setBias(false, 1)
        //
        //
        // perform training.
        //
        final int epochs = 10000;
        final double learningrate = 0.2;
        final double momentumrate = 0.95;
        //
        // generate initial weights.
        //
        net.initializeWeights(rnd, 0.1);
        //
        net.trainStochastic(
            rnd, 
            input,
            target,
            epochs,
            learningrate,
            momentumrate,
            new BasicLearningListener()
        );
        //
    }

}
