package br.cafedl.neuralnetwork.core.optimizers;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.List;

public class Adam extends Optimizer {
    private final double beta1;
    private final double beta2;
    private final double epsilon;
    private INDArray m;
    private INDArray v;
    private int t = 0;

    public Adam(double learningRate, double beta1, double beta2, double epsilon) {
        super(learningRate);
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
    }

    public Adam(double learningRate) {
        this(learningRate, 0.9, 0.999, 1e-8);
    }

    public Adam() {
        this(0.001, 0.9, 0.999, 1e-8);
    }

    public Adam(LearningRateDecayStrategy learningRateDecayStrategy, double beta1, double beta2, double epsilon) {
        super(learningRateDecayStrategy);
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
    }

    public Adam(LearningRateDecayStrategy learningRateDecayStrategy) {
        this(learningRateDecayStrategy, 0.9, 0.999, 1e-8);
    }

    /**
     * Create auxiliary parameters for Adam optimizer
     *
     * <p>
     *      m: first moment (initialized with zeros) <br>
     *      v: second moment (initialized with zeros)
     * </p>
     *
     * @param params
     * @return auxParams
     *          auxParams.get(0): m
     *          auxParams.get(1): v
     */
    @Override
    protected List<INDArray>  createAuxParams(INDArray params) {
        List<INDArray>  auxParams = new ArrayList<>(2);
        auxParams.add(Nd4j.zeros(DataType.DOUBLE, params.shape())); // m
        auxParams.add(Nd4j.zeros(DataType.DOUBLE, params.shape())); // v
        return auxParams;
    }

    /**
     * Update rule for Adam optimizer
     *
     * <p>
     *     m = beta1 * m + (1 - beta1) * grads <br>
     *     v = beta2 * v + (1 - beta2) * grads * grads <br>
     *     mHat = m / (1 - beta1^t) <br>
     *     vHat = v / (1 - beta2^t) <br>
     *     params -= learningRate * mHat / (sqrt(vHat) + epsilon) <br>
     *     t += 1 <br>
     * </p>
     *
     * @param params
     * @param grads
     * @param auxParams
     */
    @Override
    public void updateRule(INDArray params, INDArray grads, List<INDArray>  auxParams) {
        m = auxParams.get(0);
        v = auxParams.get(1);
        t++;

        m.muli(beta1).addi(grads.mul(1 - beta1));
        v.muli(beta2).addi(grads.mul(grads).mul(1 - beta2));

        INDArray mHat = m.div(1 - Math.pow(beta1, t) + epsilon);
        INDArray vHat = v.div(1 - Math.pow(beta2, t) + epsilon);
        params.subi(mHat.mul(learningRate).div(Transforms.sqrt(vHat).add(epsilon)));
    }

}
