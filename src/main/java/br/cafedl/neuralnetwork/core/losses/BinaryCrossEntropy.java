package br.cafedl.neuralnetwork.core.losses;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class BinaryCrossEntropy implements ILossFunction {
    @Override
    public INDArray forward(INDArray predictions, INDArray labels) {
        INDArray epsilon = Nd4j.scalar(1e-10);  // Para evitar log(0)
        return Transforms.log(predictions.add(epsilon)).mul(labels).add(Transforms.log(predictions.rsub(1).add(epsilon)).mul(labels.rsub(1))).sum(1).neg().mean();
    }

    @Override
    public INDArray backward(INDArray predictions, INDArray labels) {
        // -(y_true / (y_pred + epsilon) - (1 - y_true) / (1 - y_pred + epsilon)) / y_true.shape[0]
        INDArray epsilon = Nd4j.scalar(1e-10);
        return labels.div(predictions.add(epsilon)).sub(labels.rsub(1).div(predictions.rsub(1).add(epsilon))).div(labels.shape()[0]).neg();
        //return predictions.sub(labels).div(predictions.mul(predictions.rsub(1)));
    }
}
