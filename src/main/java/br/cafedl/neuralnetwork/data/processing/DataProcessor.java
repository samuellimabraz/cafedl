package br.cafedl.neuralnetwork.data.processing;

import org.nd4j.linalg.api.ndarray.INDArray;

public abstract class DataProcessor {
    public abstract void fit(INDArray data);
    public abstract INDArray transform(INDArray data);

    public abstract  INDArray inverseTransform(INDArray data);

    public INDArray fitTransform(INDArray data) {
        fit(data);
        return transform(data);
    }
}
