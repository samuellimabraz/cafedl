package br.cafedl.neuralnetwork.core.metrics;

import org.nd4j.linalg.api.ndarray.INDArray;

public class Precision implements IMetric {
    @Override
    public double evaluate(INDArray yTrue, INDArray yPred) {
        double epsilon = 1e-7; // small constant
        if (yTrue.shape()[1] == 1) {
            double truePositives = yTrue.mul(yPred).sumNumber().doubleValue();
            double predictedPositives = yPred.sumNumber().doubleValue();
            double falsePositives = predictedPositives - truePositives;
            return truePositives / (truePositives + falsePositives + epsilon);
        } else {
            int numClasses = yTrue.columns();
            double precisionAvg = 0.0;
            for (int i = 0; i < numClasses; i++) {
                INDArray yTrueClass = yTrue.getColumn(i);
                INDArray yPredClass = yPred.getColumn(i);
                double truePositives = yTrueClass.mul(yPredClass).sumNumber().doubleValue();
                double predictedPositives = yPredClass.sumNumber().doubleValue();
                double precision = truePositives / (predictedPositives + epsilon);
                precisionAvg += precision;
            }
            return precisionAvg / numClasses;
        }
    }
}

