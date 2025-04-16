package br.cafedl.neuralnetwork.core.loss;

import br.cafedl.neuralnetwork.core.losses.MeanSquaredError;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class MeanSquaredErrorTest {
    @Test
    public void testForward() {
        MeanSquaredError mse = new MeanSquaredError();
        INDArray predictions = Nd4j.create(new double[]{1, 2, 3});
        INDArray labels = Nd4j.create(new double[]{1, 2, 3});
        double expected = 0.0;
        double result = mse.forward(predictions, labels).getDouble(0);
        System.out.println(result);
        assertEquals(expected, result, 1e-7); // adicionado um pequeno delta para comparações de ponto flutuante
    }

    @Test
    public void testForward2() {
        MeanSquaredError mse = new MeanSquaredError();
        INDArray predictions = Nd4j.create(new double[]{1, 2, 3, 4, 5});
        INDArray labels = Nd4j.create(new double[]{2, 3, 4, 5, 6});
        double expected = 1.0; // A média dos quadrados das diferenças é 1
        double result = mse.forward(predictions, labels).getDouble(0);
        assertEquals(expected, result, 1e-7); // adicionado um pequeno delta para comparações de ponto flutuante
    }

    @Test
    public void testBackward() {
        MeanSquaredError mse = new MeanSquaredError();
        INDArray predictions = Nd4j.create(new double[]{1, 2, 3});
        INDArray labels = Nd4j.create(new double[]{1, 2, 3});
        INDArray expected = Nd4j.zeros(3);
        INDArray result = mse.backward(predictions, labels);

        System.out.println(result);

        assertArrayEquals(expected.toDoubleVector(), result.toDoubleVector(), 1e-6);
    }

    @Test
    public void testBackward2() {
        MeanSquaredError mse = new MeanSquaredError();
        INDArray predictions = Nd4j.create(new double[]{1, 2, 3, 4, 5});
        INDArray labels = Nd4j.create(new double[]{2, 3, 4, 5, 6});
        // O gradiente é 2*(predictions-labels)/n
        INDArray expected = Nd4j.create(new double[]{-2, -2, -2, -2, -2}).div(5);
        INDArray result = mse.backward(predictions, labels);

        System.out.println(result);

        assertArrayEquals(expected.toDoubleVector(), result.toDoubleVector(), 1e-6);
    }
}
