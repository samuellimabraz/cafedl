package br.cafedl.neuralnetwork.core.losses;

import br.cafedl.neuralnetwork.data.Util;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;


public class CategoricalCrossEntropy implements ILossFunction {
    private final double eps = 1e-8f;
    @Override
    public INDArray forward(INDArray predicted, INDArray real) {
        INDArray logprobs = real.muli(Transforms.log(Util.clip(predicted, eps, 1 - eps))).negi();
        return logprobs.sum(1).divi(real.columns());
    }

    @Override
    public INDArray backward(INDArray predicted, INDArray real) {
        return predicted.subi(real).divi(real.columns());
    }
}



