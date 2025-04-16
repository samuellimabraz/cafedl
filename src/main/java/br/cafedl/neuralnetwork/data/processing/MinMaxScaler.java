package br.cafedl.neuralnetwork.data.processing;

import org.nd4j.linalg.api.ndarray.INDArray;

public class MinMaxScaler extends DataProcessor {
    private INDArray min, max;
    private final double minRange;
    private final double maxRange;

    public MinMaxScaler(double minRange, double maxRange) {
        this.minRange = minRange;
        this.maxRange = maxRange;
    }

    public MinMaxScaler() {
        this(0, 1);
    }

    @Override
    public void fit(INDArray data) {
        min = data.min(0);
        max = data.max(0);
    }

    @Override
    public INDArray transform(INDArray data) {
        INDArray dataStd = data.subRowVector(min)
                .divRowVector(max.sub(min));
        return dataStd.mul(maxRange - minRange)
                .add(minRange);
    }

    @Override
    public INDArray inverseTransform(INDArray data) {
        return data.mul(max.sub(min))
                .add(min);
    }

}
