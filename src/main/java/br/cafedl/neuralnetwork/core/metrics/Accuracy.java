package br.cafedl.neuralnetwork.core.metrics;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Accuracy implements IMetric {
    @Override
    public double evaluate(INDArray yTrue, INDArray yPred) {
        INDArray correctPredictions = yTrue.eq(yPred).castTo(Nd4j.defaultFloatingPointType());
        return  correctPredictions.meanNumber().doubleValue();
    }
}
