package br.cafedl.neuralnetwork.data.processing;

import org.nd4j.linalg.api.ndarray.INDArray;

public class StandardScaler extends DataProcessor {
    private double mean, std;
    private static final double EPSILON = 1e-8;

    @Override
    public void fit(INDArray data) {
        mean = data.meanNumber().doubleValue();
        std = data.stdNumber().doubleValue(); // Adiciona suavização ao desvio padrão
    }

    @Override
    public INDArray transform(INDArray data) {
        return data.subi(mean).div(std + EPSILON);
    }

    @Override
    public INDArray inverseTransform(INDArray data) {
        return data.mul(std).add(mean);
    }

    public double getMean() {
        return mean;
    }

    public double getStd() {
        return std;
    }
}
