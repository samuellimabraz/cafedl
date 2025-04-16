package br.cafedl.neuralnetwork.core.activation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class LeakyReLU implements IActivation {
    private double alpha = 0.05;

    @Override
    public INDArray forward(INDArray input) {
        return Transforms.max(input.mul(alpha), input, false);
    }

    @Override
    public INDArray backward(INDArray input) {
        // Derivative of LeakyReLU(x) = 1 for x > 0 and 0.01 for x <= 0
        INDArray positiveMask = input.gt(0);
        INDArray negativeMask = input.lte(0);
        return Nd4j.onesLike(input).mul(positiveMask).add(Nd4j.onesLike(input).mul(alpha).mul(negativeMask));
    }

    public void setAlpha(double alpha) {
        this.alpha = alpha;
    }
}
