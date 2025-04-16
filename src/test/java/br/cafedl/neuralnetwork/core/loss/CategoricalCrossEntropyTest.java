package br.cafedl.neuralnetwork.core.loss;

import br.cafedl.neuralnetwork.core.losses.CategoricalCrossEntropy;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class CategoricalCrossEntropyTest {
    @Test
    public void testForward() {
        CategoricalCrossEntropy loss = new CategoricalCrossEntropy();
        INDArray predicted = Nd4j.create(new float[]{0.4f, 0.5f, 0.1f}, new int[]{1, 3});
        INDArray real = Nd4j.create(new float[]{0f, 1f, 0f}, new int[]{1, 3});
        INDArray result = loss.forward(predicted, real);
        float expected = 0.6931f / 3;

        System.out.println("result: " + result);
        System.out.println("expected: " + expected);

        assertEquals(expected, result.getFloat(0), 1e-4);
    }

    @Test
    public void testBackward() {
        CategoricalCrossEntropy loss = new CategoricalCrossEntropy();
        INDArray predicted = Nd4j.create(new float[]{0.4f, 0.5f, 0.1f}, new int[]{1, 3});
        INDArray real = Nd4j.create(new float[]{0f, 1f, 0f}, new int[]{1, 3});
        INDArray result = loss.backward(predicted, real);
        INDArray expected = Nd4j.create(new float[]{0.4f, -0.5f, 0.1f}, new int[]{1, 3});
        expected.divi(3);

        System.out.println("result: " + result);
        System.out.println("expected: " + expected);

        assertArrayEquals(expected.toFloatVector(), result.toFloatVector(), 1e-4F);
    }
}
