package br.cafedl.neuralnetwork.core.activation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * Sigmoid activation function
 */
public class Sigmoid implements IActivation {

    /**
     * Apply sigmoid activation function to ndarray input
     * @param input (INDArray)
     * @return 1 / (1 + e^(-input)) (INDArray)
     */
    @Override
    public INDArray forward(INDArray input) {
        return Transforms.exp(input.neg()).add(1.0).rdiv(1);
    }

    /**
     * Calculate derivative of sigmoid activation function
     * @param input (INDArray)
     * @return sigmoid(input) * (1 - sigmoid(input)) (INDArray)
     */
    @Override
    public INDArray backward(INDArray input) {
        INDArray sigmoid = forward(input);
        return sigmoid.mul(sigmoid.rsub(1.0));
    }
}
