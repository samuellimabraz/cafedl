package br.cafedl.neuralnetwork.core.metrics;

import org.nd4j.linalg.api.ndarray.INDArray;

public class F1Score implements IMetric {
    @Override
    public double evaluate(INDArray yTrue, INDArray yPred) {
        double precision = new Precision().evaluate(yTrue, yPred);
        double recall = new Recall().evaluate(yTrue, yPred);
        return 2 * (precision * recall) / (precision + recall + 1e-7);
    }
}
