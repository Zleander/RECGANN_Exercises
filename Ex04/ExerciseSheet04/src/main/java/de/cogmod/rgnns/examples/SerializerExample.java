package de.cogmod.rgnns.examples;

import java.io.IOException;
import java.util.Random;

import de.jannlab.io.Serializer;

/**
 * @author Sebastian Otte
 */
public class SerializerExample {

	public static void main(String[] args) {

    	final double[] values = new double[10];
    	final Random rnd = new Random(1234);
    	
    	for (int i = 0; i < values.length; i++) {
    		values[i] = rnd.nextDouble();
    	}
    	
    	//
    	// Write double array into file.
    	//
		try {
			Serializer.write(values, "data/test.dat");
		} catch (IOException e) {
			e.printStackTrace();
		}
		//
		// Read double array from file.
		//
		try {
			final double[] values2 = Serializer.read("data/test.dat");
			
			for (int i = 0; i < values2.length; i++) {
				System.out.println(values2[i]);
			}
			
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
    	System.out.println();

    }  
}