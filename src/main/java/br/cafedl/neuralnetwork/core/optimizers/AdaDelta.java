package br.cafedl.neuralnetwork.core.optimizers;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.List;

public class AdaDelta extends Optimizer{
    private final double decayRate;
    private final double epsilon;
    private INDArray accumulator;
    private INDArray delta;

    public AdaDelta(double decayRate, double epsilon) {
        this.decayRate = decayRate;
        this.epsilon = epsilon;
    }

    public AdaDelta(double decayRate) {
        this(decayRate, 10e-5);
    }

    public AdaDelta() {
        this(0.95, 10e-5);
    }

    /**
     * Create the auxiliary parameters of the model
     * <p>
     * accumulator = zeros(params.shape) <br>
     * delta = zeros(params.shape)
     * <p>
     */
    @Override
    protected List<INDArray> createAuxParams(INDArray params) {
        List<INDArray>  auxParams = new ArrayList<>(2);
        auxParams.add(Nd4j.zeros(DataType.DOUBLE, params.shape())); // accumulator
        auxParams.add(Nd4j.zeros(DataType.DOUBLE, params.shape())); // delta
        return auxParams;
    }

    /**
     * Update the parameters of the model
     * <p>
     * accumulator = decayRate * accumulator + (1 - decayRate) * grads^2 <br>
     * update = sqrt(delta + epsilon) / sqrt(accumulator + epsilon) * grads <br>
     * params = params - update <br>
     * delta = decayRate * delta + (1 - decayRate) * update^2
     * <p>
     */
    @Override
    public void updateRule(INDArray params, INDArray grads, List<INDArray>  auxParams) {
        accumulator = auxParams.get(0);
        delta = auxParams.get(1);

        delta.muli(decayRate).addi(Transforms.pow(grads, 2).mul(1 - decayRate));
        INDArray update = Transforms.sqrt(accumulator.add(epsilon)).divi(Transforms.sqrt(delta.add(epsilon))).muli(grads);
        accumulator.muli(decayRate).addi(Transforms.pow(update, 2).muli(1 - decayRate));
        params.subi(update);
    }
}
