package br.cafedl.neuralnetwork.core.optimizers;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.List;

public class AdaGrad extends Optimizer {
    private final double eps = 1e-7;

    private INDArray sumSquares = null;

    public AdaGrad(double lr) {
        super(lr);
    }

    public AdaGrad() {
        this(0.01);
    }

    public AdaGrad(LearningRateDecayStrategy learningRateDecayStrategy) {
        super(learningRateDecayStrategy);
    }

    @Override
    protected List<INDArray> createAuxParams(INDArray params) {
        List<INDArray>  auxParams = new ArrayList<>(1);
        auxParams.add(Nd4j.zeros(DataType.DOUBLE, params.shape())); // sumSquares
        return auxParams;
    }

    /**
     * Update the parameters of the model
     * <p>
     * sumSquares = sumSquares + grads^2 <br>
     * params = params - learningRate * grads / sqrt(sumSquares + epsilon)
     * <p>
     */
    @Override
    public void updateRule(INDArray params, INDArray grads, List<INDArray>  auxParams) {
        sumSquares = auxParams.get(0);
        sumSquares.addi(Transforms.pow(grads, 2));
        params.subi(grads.mul(learningRate).div(Transforms.sqrt(sumSquares).add(eps)));
    }
}