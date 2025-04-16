package br.cafedl.neuralnetwork.core.activation;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Arrays;


public class ActivationTest {

    @Test
    public void testEnumCreate() {
        IActivation sigmoid = Activation.create(ActivateEnum.SIGMOID);
        assertTrue(sigmoid instanceof Sigmoid);

        IActivation tanh = Activation.create(ActivateEnum.TANH);
        assertTrue(tanh instanceof TanH);

        IActivation relu = Activation.create(ActivateEnum.RELU);
        assertTrue(relu instanceof ReLU);

        IActivation softmax = Activation.create(ActivateEnum.SOFTMAX);
        assertTrue(softmax instanceof Softmax);

        IActivation silu = Activation.create(ActivateEnum.SILU);
        assertTrue(silu instanceof SiLU);

        IActivation linear = Activation.create(ActivateEnum.LINEAR);
        assertTrue(linear instanceof Linear);
    }

    @Test
    public void testLabelCreate() {
        IActivation sigmoid = Activation.create("sigmoid");
        assertTrue(sigmoid instanceof Sigmoid);

        IActivation tanh = Activation.create("tanh");
        assertTrue(tanh instanceof TanH);

        IActivation relu = Activation.create("relu");
        assertTrue(relu instanceof ReLU);

        IActivation softmax = Activation.create("softmax");
        assertTrue(softmax instanceof Softmax);

        IActivation silu = Activation.create("silu");
        assertTrue(silu instanceof SiLU);

        IActivation linear = Activation.create("linear");
        assertTrue(linear instanceof Linear);
    }

    @Test
    public void testNDimensions() {
        // Crie um array de ativações para testar
        IActivation[] activations = new IActivation[]{
                Activation.create(ActivateEnum.SIGMOID),
                Activation.create(ActivateEnum.TANH),
                Activation.create(ActivateEnum.RELU),
                Activation.create(ActivateEnum.SOFTMAX),
                Activation.create(ActivateEnum.SILU),
                Activation.create(ActivateEnum.LINEAR)
        };

        // Para cada ativação
        for (IActivation activation : activations) {
            // Crie um tensor de entrada de 3 dimensões
            INDArray input = Nd4j.rand(2, 3, 4).sub(0.5).mul(2);

            System.out.println("Activation: " + activation.getClass().getSimpleName());

            // Aplique a função de ativação
            INDArray outputForward = activation.forward(input);
            INDArray outputBackward = activation.backward(input);

            // Verifique se o formato do tensor de saída é o mesmo que o do tensor de entrada
            assertArrayEquals(input.shape(), outputForward.shape());
            assertArrayEquals(input.shape(), outputBackward.shape());
        }
    }


    @Test
    public void testSigmoid() {
        IActivation sigmoid = Activation.create(ActivateEnum.SIGMOID);
        INDArray input = Nd4j.create(new float[]{0, 1, -1});
        INDArray expectedForward = Nd4j.create(new float[]{0.5f, 0.7310586f, 0.26894143f});
        INDArray expectedBackward = Nd4j.create(new float[]{0.25f, 0.19661194f, 0.19661194f});
        assertTrue(sigmoid.forward(input).equalsWithEps(expectedForward, 1e-6f));
        assertTrue(sigmoid.backward(input).equalsWithEps(expectedBackward, 1e-6f));
    }

    @Test
    public void testTanH() {
        IActivation tanh = Activation.create(ActivateEnum.TANH);
        INDArray input = Nd4j.create(new float[]{0, 1, -1}, new int[]{1, 3});
        INDArray expectedOutput = Nd4j.create(new float[]{0, 0.7615942f, -0.7615942f}, new int[]{1, 3});
        assertTrue(tanh.forward(input).equalsWithEps(expectedOutput, 1e-6f));
    }

    @Test
    public void testReLU() {
        IActivation relu = Activation.create(ActivateEnum.RELU);
        INDArray input = Nd4j.create(new float[]{0, 1, -1, 5});
        INDArray expectedForward = Nd4j.create(new float[]{0, 1, 0, 5});
        INDArray expectedBackward = Nd4j.create(new float[]{0, 1, 0, 1});
        System.out.println("relu.forward(input) = " + relu.forward(input));
        System.out.println("relu.backward(input) = " + relu.backward(input));
        assertTrue(relu.forward(input).equalsWithEps(expectedForward, 1e-6f));
        assertTrue(relu.backward(input).equalsWithEps(expectedBackward, 1e-6f));
    }

    @Test
    public void testSoftmax() {
        IActivation softmax = Activation.create(ActivateEnum.SOFTMAX);
        INDArray input = Nd4j.create(new float[][]{{1, 2, 3}, {1, 2, 3}, {4, 2, 3}});
        INDArray expectedForward = Transforms.softmax(input);
        INDArray expectedBackward = expectedForward.mul((Nd4j.onesLike(expectedForward).sub(expectedForward)));

        INDArray outputForward = softmax.forward(input);
        INDArray outputBackward = softmax.backward(input);

        System.out.println("Input shape: " + Arrays.toString(input.shape()));
        System.out.println("Output shape: " + Arrays.toString(outputForward.shape()));

        assertArrayEquals(input.shape(), outputForward.shape());
        assertArrayEquals(input.shape(), outputBackward.shape());

        System.out.println("Softmax = " + expectedForward);
        System.out.println("Output: " + outputForward);
        System.out.println("Sum = " + outputForward.sum(1));

        System.out.println("Softamax' = " + expectedBackward);
        System.out.println("Output: " + outputBackward);


        assertTrue(outputForward.equalsWithEps(expectedForward, 1e-6f));
        assertTrue(outputBackward.equalsWithEps(expectedBackward, 1e-6f));
    }

    @Test
    public void testLinear() {
        IActivation linear = Activation.create(ActivateEnum.LINEAR);
        INDArray input = Nd4j.create(new float[]{0, 1, -1, 5});
        INDArray expectedForward = Nd4j.create(new float[]{0, 1, -1, 5});
        INDArray expectedBackward = Nd4j.create(new float[]{1, 1, 1, 1});
        assertTrue(linear.forward(input).equalsWithEps(expectedForward, 1e-6f));
        assertTrue(linear.backward(input).equalsWithEps(expectedBackward, 1e-6f));
    }

    @Test
    public void testSiLU() {
        IActivation silu = Activation.create(ActivateEnum.SILU);

        INDArray input = Nd4j.create(new float[]{0, 1, -1, 5});
        INDArray sigmoid = Transforms.sigmoid(input);

        INDArray expectedForward = input.mul(sigmoid);
        INDArray expectedBackward = sigmoid.add(input.mul(sigmoid).mul(Nd4j.onesLike(input).sub(sigmoid)));

        assertTrue(silu.forward(input).equalsWithEps(expectedForward, 1e-6f));
        assertTrue(silu.backward(input).equalsWithEps(expectedBackward, 1e-6f));
    }
}
