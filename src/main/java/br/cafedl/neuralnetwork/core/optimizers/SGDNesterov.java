package br.cafedl.neuralnetwork.core.optimizers;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

public class SGDNesterov extends Optimizer {
    private final double momentum;
    private INDArray velocities;

    public SGDNesterov(double learningRate, double momentum) {
        super(learningRate);
        this.momentum = momentum;
    }

    public SGDNesterov(double learningRate) {
        this(learningRate, 0.9);
    }

    public SGDNesterov(LearningRateDecayStrategy learningRateDecayStrategy, double momentum) {
        super(learningRateDecayStrategy);
        this.momentum = momentum;
    }

    public SGDNesterov(LearningRateDecayStrategy learningRateDecayStrategy) {
        this(learningRateDecayStrategy, 0.9);
    }

    /**
     * Create auxiliary parameters for SGDNesterov optimizer
     *
     * <p>
     *     velocities: velocities (initialized with zeros)
     * </p>
     * @param params
     * @return auxParams
     *         auxParams.get(0): velocities
     */
    @Override
    protected List<INDArray> createAuxParams(INDArray params) {
        List<INDArray>  auxParams = new ArrayList<>(1);
        auxParams.add(Nd4j.zeros(params.shape()));
        return auxParams;
    }

    /**
     * Update rule for SGDNesterov optimizer
     *
     * <p>
     *     velocities = momentum * velocities - learningRate * grads <br>
     *     velocities = momentum * velocities - learningRate * grads <br>
     *     params = params + velocities
     * </p>
     *
     * @param params
     * @param grads
     * @param auxParams
     *         auxParams.get(0): velocities
     */
    @Override
    public void updateRule(INDArray params, INDArray grads, List<INDArray>  auxParams) {
        velocities = auxParams.get(0);
        velocities.muli(momentum).addi(grads.mul(-learningRate));
        velocities.muli(momentum).addi(grads.mul(-learningRate));
        params.addi(velocities);
    }
}
