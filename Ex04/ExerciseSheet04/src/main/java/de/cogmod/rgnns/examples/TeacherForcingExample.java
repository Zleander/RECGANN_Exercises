package de.cogmod.rgnns.examples;

import java.util.Random;

import de.cogmod.rgnns.EchoStateNetwork;


/**
 * @author Sebastian Otte
 */
public class TeacherForcingExample {
   
    public static void main(String[] args) {
        //
        final Random rnd = new Random(1234);
        //
        // We generate a random (untrained) oscillator ESN.
        // Note that internally the output feedback is realized 
        // with the input layer. Consequently, the input weights
        // can be seen as the output feedback weights and any OFB
        // scaling should be applied to them.
        //
        final EchoStateNetwork esn = new EchoStateNetwork(1, 3, 1);
        esn.initializeWeights(new Random(1234), 0.1);
        //
        // In this example the ESN's dynamics are initialized
        // with 10 washout iterations (via teacher forcing).
        //
        // washout with teacher forcing
        //
        final double[] target = new double[1];
        //
        System.out.println("washout with teacher forcing");
        //
        for (int t = 0; t < 10; t++) {
            //
            // Perform an oscillator forward pass
            // in which the esn is fed with its 
            // previous output (note that during 
            // teacher forcing the output of the ESN
            // is overwritten with the target) 
            //
            final double[] output = esn.forwardPassOscillator();
            System.out.println(output[0]);
            //
            // Generate a exemplary random target.
            //
            target[0] = rnd.nextGaussian();
            //
            // The semantic here is as follows: 
            //
            // o We assume that the current target value
            //   is the desired output of the current
            //   calculation step of the ESN -> thus,
            //   we want that ideally output is the 
            //   same as target. 
            // 
            // o Via teacher forcing,
            //   we enforce the ESN to unfold its 
            //   dynamics as it would have produced
            //   target as output.
            //
            esn.teacherForcing(target);
        }
        // 
        // We now look at the output of the ESN that is produces
        // really driven by its own output.
        //
        System.out.println();
        System.out.println("output feedback");
        //
        for (int t = 0; t < 20; t++) {
            final double[] output = esn.forwardPassOscillator();
            System.out.println(output[0]);
        }
    }
 
    
}