package br.cafedl.neuralnetwork.core.activation;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * ReLU's activation function
 */
public class ReLU implements IActivation {

    /**
     * Apply ReLU activation function to ndarray input
     * @param input (INDArray)
     * @return max(0, input) (INDArray)
     */
    @Override
    public INDArray forward(INDArray input) {
        INDArray mask = input.gt(0);  // Create a boolean mask where the input is greater than 0
        return input.muli(mask);  // Multiply the input by the mask
    }

    /**
     * Calculate derivative of ReLU activation function
     * @param input (INDArray)
     * @return 1 if input > 0, 0 otherwise (INDArray)
     */
    @Override
    public INDArray backward(INDArray input) {
        return input.gt(0).castTo(input.dataType());  // Derivative of ReLU is 1 if input > 0, 0 otherwise
    }
}
