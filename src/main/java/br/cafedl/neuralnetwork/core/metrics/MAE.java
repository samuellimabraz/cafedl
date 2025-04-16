package br.cafedl.neuralnetwork.core.metrics;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

public class MAE implements IMetric {
    @Override
    public double evaluate(INDArray yTrue, INDArray yPred) {
        return Transforms.abs(yTrue.sub(yPred)).meanNumber().doubleValue();
    }
}
