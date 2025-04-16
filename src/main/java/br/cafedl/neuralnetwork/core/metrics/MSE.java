package br.cafedl.neuralnetwork.core.metrics;

import org.nd4j.linalg.api.ndarray.INDArray;

public class MSE implements IMetric {
    @Override
    public double evaluate(INDArray yTrue, INDArray yPred) {
        return yTrue.sub(yPred).mul(yTrue.sub(yPred)).meanNumber().doubleValue();
    }
}
