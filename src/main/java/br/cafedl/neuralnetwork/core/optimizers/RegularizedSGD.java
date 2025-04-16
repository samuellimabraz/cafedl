package br.cafedl.neuralnetwork.core.optimizers;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;

public class RegularizedSGD extends Optimizer {
    private final double alpha;

    public RegularizedSGD(double learningRate, double alpha) {
        super(learningRate);
        this.alpha = alpha;
    }

    public RegularizedSGD() {
        this(0.01, 0.1);
    }

    public RegularizedSGD(double learningRate) {
        this(learningRate, 0.1);
    }

    public RegularizedSGD(LearningRateDecayStrategy learningRateDecayStrategy) {
        super(learningRateDecayStrategy);
        this.alpha = 0.1;
    }

    public RegularizedSGD(LearningRateDecayStrategy learningRateDecayStrategy, double alpha) {
        super(learningRateDecayStrategy);
        this.alpha = alpha;
    }

    @Override
    protected List<INDArray>  createAuxParams(INDArray params) {
        return null;
    }

    @Override
    public void updateRule(INDArray params, INDArray grads, List<INDArray>  auxParams) {
        params.subi(grads.mul(learningRate).add(params.mul(alpha)));
    }
}
