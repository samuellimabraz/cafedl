package br.cafedl.neuralnetwork.core.optimizers;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;

public class SGD extends Optimizer {
    public SGD(double learningRate) {
        super(learningRate);
    }

    public SGD(LearningRateDecayStrategy learningRateDecayStrategy) {
        super(learningRateDecayStrategy);
    }

    public SGD() {
        this(0.001);
    }

    /**
     * Create auxiliary parameters for SGD optimizer
     *
     * <p>
     *     No auxiliary parameters are needed for SGD
     *     so this method returns null
     * </p>
     * @param params
     * @return auxParams = null
     */
    @Override
    protected List<INDArray> createAuxParams(INDArray params) {
        return null;
    }

    /**
     * Update rule for SGD optimizer
     *
     * <p>
     *     params = params - learningRate * grads <br>
     * </p>
     * @param params
     * @param grads
     * @param auxParams
     */
    @Override
    public void updateRule(INDArray params, INDArray grads, List<INDArray> auxParams) {
        params.subi(grads.mul(learningRate));
    }

}