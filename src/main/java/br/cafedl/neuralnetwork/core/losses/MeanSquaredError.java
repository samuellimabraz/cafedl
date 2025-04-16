package br.cafedl.neuralnetwork.core.losses;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

public class MeanSquaredError implements ILossFunction {
    @Override
    public INDArray forward(INDArray predictions, INDArray labels) {
        return Transforms.pow(predictions.sub(labels), 2).mean();
    }

    @Override
    public INDArray backward(INDArray predictions, INDArray labels) {
        return predictions.sub(labels).mul(2).div(predictions.shape()[0]);
    }
}
