package br.cafedl.neuralnetwork.core.losses;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface ILossFunction {
    INDArray forward(INDArray predicted, INDArray real);
    INDArray backward(INDArray predicted, INDArray real);
}
