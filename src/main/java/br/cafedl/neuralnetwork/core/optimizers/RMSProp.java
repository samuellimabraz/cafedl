package br.cafedl.neuralnetwork.core.optimizers;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.*;

/**
 * RMSProp optimizer
 */
public class RMSProp extends Optimizer {
    private final double decayRate;
    private final double epsilon;
    private INDArray accumulator;

    public RMSProp(double learningRate, double decayRate, double epsilon) {
        super(learningRate);
        this.learningRate = learningRate;
        this.decayRate = decayRate;
        this.epsilon = epsilon;
    }

    public RMSProp(LearningRateDecayStrategy learningRateDecayStrategy, double decayRate, double epsilon) {
        super(learningRateDecayStrategy);
        this.decayRate = decayRate;
        this.epsilon = epsilon;
    }

    public RMSProp(LearningRateDecayStrategy learningRateDecayStrategy) {
        this(learningRateDecayStrategy, 0.9, 1e-8);
    }

    public RMSProp() {
        this(0.001, 0.9, 1e-8);
    }

    public RMSProp(double learningRate, double decayRate) {
        this(learningRate, decayRate, 1e-8);
    }

    public RMSProp(double learningRate) {
        this(learningRate, 0.9, 1e-8);
    }

    @Override
    protected List<INDArray>  createAuxParams(INDArray params) {
        List<INDArray>  auxParams = new ArrayList<>(1);
        auxParams.add(Nd4j.zeros(params.shape()).castTo(DataType.DOUBLE)); // accumulator
        return auxParams;
    }

    /**
     * Update the parameters of the model
     * <p>
     * accumaletor = decayRate * accumaletor + (1 - decayRate) * grads^2 <br>
     * params = params - learningRate * grads / sqrt(accumaletor + epsilon)
     * <p>
     */
    @Override
    protected void updateRule(INDArray params, INDArray grads, List<INDArray> auxParams) {
        accumulator = auxParams.get(0);
        accumulator.muli(decayRate).addi(Transforms.pow(grads, 2).mul(1 - decayRate));
        params.subi(grads.muli(learningRate).divi(Transforms.sqrt(accumulator).addi(epsilon)));
    }

}
