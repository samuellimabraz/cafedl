package br.cafedl.neuralnetwork.core.layers;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class ZeroPadding2DTest {

    @Test
    public void testForwardAndBackward() {
        int padding = 2;
        ZeroPadding2D zeroPadding2D = new ZeroPadding2D(padding);

        INDArray input = Nd4j.randn(4, 3, 3, 2);
        INDArray output = zeroPadding2D.forward(input.dup());

        System.out.println("Input:" + input);
        System.out.println("Output:" + output);

        assertEquals(input.size(0), output.size(0)); // batch size
        assertEquals(input.size(1) + 2 * padding, output.size(1)); // height
        assertEquals(input.size(2) + 2 * padding, output.size(2)); // width
        assertEquals(input.size(3), output.size(3)); // channels

        INDArray gradInput = zeroPadding2D.backward(output.dup());

        System.out.println("Backward: " + gradInput);

        assertTrue(Arrays.equals(input.shape(), gradInput.shape()));
    }
}
