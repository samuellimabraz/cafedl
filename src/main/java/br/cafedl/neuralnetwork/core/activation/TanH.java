package br.cafedl.neuralnetwork.core.activation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * TanH activation function.
 * 
 */
public class TanH implements IActivation {

    /**
     * Applies the TanH function to the input. Use Transforms.tanh() from ND4J.
     * 
     * @param input (INDArray)
     * @return tanh(input) (INDArray)
     */
    @Override
    public INDArray forward(INDArray input) {
        return Transforms.tanh(input);
    }

    /**
     * Calculate derivative of TanH activation function.
     * 
     * @param input (INDArray)
     * @return 1 - tanh(input)^2 (INDArray)
     */
    @Override
    public INDArray backward(INDArray input) {
        INDArray tanh = forward(input);
        return Transforms.pow(tanh, 2).rsub(1);
    }
}
