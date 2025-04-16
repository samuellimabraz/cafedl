package br.cafedl.neuralnetwork.core.metrics;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

public class RMSE implements IMetric {
    @Override
    public double evaluate(INDArray yTrue, INDArray yPred) {
        return Math.sqrt(Transforms.pow(yTrue.sub(yPred), 2).meanNumber().doubleValue());
    }
}
