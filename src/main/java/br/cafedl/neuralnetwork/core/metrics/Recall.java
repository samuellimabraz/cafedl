package br.cafedl.neuralnetwork.core.metrics;

import org.nd4j.linalg.api.ndarray.INDArray;

public class Recall implements IMetric {
    @Override
    public double evaluate(INDArray yTrue, INDArray yPred) {
        double epsilon = 1e-7; // small constant
        if (yTrue.shape()[1] == 1) {
            double truePositives = yTrue.mul(yPred).sumNumber().doubleValue();
            double actualPositives = yTrue.sumNumber().doubleValue();
            double falseNegatives = actualPositives - truePositives;
            return truePositives / (truePositives + falseNegatives + epsilon);
        } else {
            int numClasses = yTrue.columns();
            double recallAvg = 0.0;
            for (int i = 0; i < numClasses; i++) {
                INDArray yTrueClass = yTrue.getColumn(i);
                INDArray yPredClass = yPred.getColumn(i);
                double truePositives = yTrueClass.mul(yPredClass).sumNumber().doubleValue();
                double actualPositives = yTrueClass.sumNumber().doubleValue();
                double recall = truePositives / (actualPositives + epsilon);
                recallAvg += recall;
            }
            return recallAvg / numClasses;
        }
    }
}
