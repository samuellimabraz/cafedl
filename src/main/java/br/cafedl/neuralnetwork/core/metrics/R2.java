package br.cafedl.neuralnetwork.core.metrics;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * R2 metric
 * R2 = 1 - SSres / SStot
 * SSres = sum((yTrue - yPred)^2)
 * SStot = sum((yTrue - mean(yTrue))^2)
 */
public class R2 implements IMetric {
    @Override
    public double evaluate(INDArray yTrue, INDArray yPred) {
        INDArray ssRes = yTrue.sub(yPred).mul(yTrue.sub(yPred)).sum();
        INDArray ssTot = yTrue.sub(yTrue.mean()).mul(yTrue.sub(yTrue.mean())).sum();
        return 1 - ssRes.div(ssTot).getDouble(0);
    }
}
