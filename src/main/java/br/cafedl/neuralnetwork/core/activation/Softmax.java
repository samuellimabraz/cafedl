package br.cafedl.neuralnetwork.core.activation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * Softmax's activation function
 */
public class Softmax implements IActivation {
    /**
     * Apply softmax activation function to ndarray input
     * @param input (INDArray)
     * @return exp(x - max(x)) / sum(exp(x - max(x))) (INDArray)
     */
    @Override
    public INDArray forward(INDArray input) {
        return Transforms.softmax(input);
    }

    /**
     * Calculate derivative of softmax activation function
     * @param input (INDArray)
     * @return softmax(input) * (1 - softmax(input)) (INDArray)
     */
    @Override
    public INDArray backward(INDArray input) {
        INDArray softmax = this.forward(input);
        return softmax.muli(Nd4j.onesLike(softmax).subi(softmax)).reshape(input.shape());
    }
}
