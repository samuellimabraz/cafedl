package br.cafedl.neuralnetwork.core.layers;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

public class MaxPooling2DTest {

    @Test
    public void testForwardAndBackward() {
        // Initialize a MaxPooling2D layer
        MaxPooling2D maxPooling2D = new MaxPooling2D(2, 2);

        // Create a 4D input tensor (for simplicity, we use a 1x4x4x1 tensor)
        INDArray input = Nd4j.create(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, new int[]{1, 4, 4, 1});

        // Forward pass
        INDArray output = maxPooling2D.forward(input);

        // Check the shape of the output
        long[] longShape = output.shape();
        int[] intShape = Arrays.stream(longShape).mapToInt(i -> (int) i).toArray();

        System.out.println("Input shape:" + input.shapeInfoToString());
        System.out.println("Input:" + input);
        System.out.println("Output Shape:" + output.shapeInfoToString());
        System.out.println("Output:" + output);

        assertArrayEquals(new int[]{1, 2, 2, 1}, intShape);

        // Check the values of the output
        assertEquals(6.0, output.getDouble(0, 0, 0, 0), "Incorrect max pooling result");
        assertEquals(8.0, output.getDouble(0, 0, 1, 0), "Incorrect max pooling result");
        assertEquals(14.0, output.getDouble(0, 1, 0, 0), "Incorrect max pooling result");
        assertEquals(16.0, output.getDouble(0, 1, 1, 0), "Incorrect max pooling result");

        // Backward pass
        System.out.println(" * --------------------- BACKWARD PASS --------------------- * ");
        INDArray gradout = Nd4j.ones(1, 2, 2, 1); // for simplicity, we use a tensor filled with ones
        INDArray grad = maxPooling2D.backward(gradout);

        System.out.println("Gradout shape:" + gradout.shapeInfoToString());
        System.out.println("Gradout:" + gradout);

        System.out.println("Grad Shape:" + grad.shapeInfoToString());
        System.out.println("Grad:" + grad);

        // Check the shape of the gradient
        assertArrayEquals(input.shape(), grad.shape());

        // Check the values of the gradient
        assertEquals(0.0, grad.getDouble(0, 0, 0, 0), "Incorrect gradient result");
        assertEquals(0.0, grad.getDouble(0, 0, 1, 0), "Incorrect gradient result");
        assertEquals(0.0, grad.getDouble(0, 0, 2, 0), "Incorrect gradient result");
        assertEquals(0.0, grad.getDouble(0, 0, 3, 0), "Incorrect gradient result");
        assertEquals(0.0, grad.getDouble(0, 1, 0, 0), "Incorrect gradient result");
        assertEquals(1.0, grad.getDouble(0, 1, 1, 0), "Incorrect gradient result");
        assertEquals(0.0, grad.getDouble(0, 1, 2, 0), "Incorrect gradient result");
        assertEquals(1.0, grad.getDouble(0, 1, 3, 0), "Incorrect gradient result");
        assertEquals(0.0, grad.getDouble(0, 2, 0, 0), "Incorrect gradient result");
        assertEquals(0.0, grad.getDouble(0, 2, 1, 0), "Incorrect gradient result");
        assertEquals(0.0, grad.getDouble(0, 2, 2, 0), "Incorrect gradient result");
        assertEquals(0.0, grad.getDouble(0, 2, 3, 0), "Incorrect gradient result");
        assertEquals(0.0, grad.getDouble(0, 3, 0, 0), "Incorrect gradient result");
        assertEquals(1.0, grad.getDouble(0, 3, 1, 0), "Incorrect gradient result");
        assertEquals(0.0, grad.getDouble(0, 3, 2, 0), "Incorrect gradient result");
        assertEquals(1.0, grad.getDouble(0, 3, 3, 0), "Incorrect gradient result");
    }
}
