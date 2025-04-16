package br.cafedl.neuralnetwork.core.layers;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

public class FlattenTest {

    @Test
    public void testForward() {
        // Create an instance of Flatten
        Flatten flatten = new Flatten();

        // Create some input data
        INDArray inputs = Nd4j.rand(1, 5, 5, 3); // 1 image, 5x5 size, 3 channels

        // Call the forward method
        INDArray output = flatten.forward(inputs);

        System.out.println("Output:" + output);
        System.out.println("Output Shape:" + output.shapeInfoToString());

        // Check the shape of the output
        long[] expectedShape = {1, 5*5*3};
        assertArrayEquals(expectedShape, output.shape());
    }

    @Test
    public void testBackward() {
        // Create an instance of Flatten
        Flatten flatten = new Flatten();

        // Create some input data
        INDArray inputs = Nd4j.rand(1, 5, 5, 3); // 1 image, 5x5 size, 3 channels

        System.out.println("Input:" + inputs);

        // Call the forward method
        INDArray output = flatten.forward(inputs);

        System.out.println("Output:" + output);
        System.out.println("Output Shape:" + output.shapeInfoToString());

        // Call the backward method
        INDArray gradInput = flatten.backward(output);

        System.out.println("GradInput:" + gradInput);
        System.out.println("GradInput Shape:" + gradInput.shapeInfoToString());

        // Check the shape of the gradient input
        assertArrayEquals(inputs.shape(), gradInput.shape());
    }
}
