package br.cafedl.neuralnetwork.core.layers;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.*;

public class DropoutTest {

    @Test
    public void testForwardAndBackward() {
        // Create a Dropout object with a dropout rate of 0.5
        Dropout dropout = new Dropout(0.2);

        // Create a mock input
        INDArray input = Nd4j.ones(1, 5);

        System.out.println("input: " + input);

        // Call the forward method
        INDArray output = dropout.forward(input);

        System.out.println("input: " + input);
        System.out.println("output: " + output);

        // Verify the output
        assertTrue(output.sumNumber().doubleValue() <= input.sumNumber().doubleValue());

        // Create a mock gradout
        INDArray gradout = Nd4j.ones(5, 5);

        // Call the backward method
        INDArray backOutput = dropout.backward(gradout);

        // Verify the backOutput
        assertTrue(backOutput.sumNumber().doubleValue() <= gradout.sumNumber().doubleValue());
    }
}