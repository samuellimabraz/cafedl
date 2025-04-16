package br.cafedl.neuralnetwork.core.losses;

import br.cafedl.neuralnetwork.core.activation.Softmax;
import br.cafedl.neuralnetwork.data.Util;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

public class SoftmaxCrossEntropy implements ILossFunction {
    private double eps = 1e-9;
    private boolean singleClass = false;
    private INDArray softmaxPreds;

    private final Softmax softmax = new Softmax();

    public  SoftmaxCrossEntropy() {}

    public SoftmaxCrossEntropy(double eps) {
        this.eps = eps;
    }

    @Override
    public INDArray forward(INDArray predicted, INDArray real) {
        if (real.shape()[1] == 0) {
            singleClass = true;
            System.out.println("Single class");
        }

        if (singleClass) {
            predicted = Util.normalize(predicted);
            real = Util.normalize(real);
        }

        this.softmaxPreds = Util.clip(softmax.forward(predicted), eps, 1 - eps);

        INDArray softmaxCrossEntropyLoss = real.mul(-1).mul(Transforms.log(softmaxPreds)).subi(
                real.rsub(1).mul(Transforms.log(softmaxPreds.rsub(1))));

        if (singleClass) {
            return softmaxCrossEntropyLoss.sum();
        } else {
            return softmaxCrossEntropyLoss.sum().divi(real.shape()[1]);
        }
    }

    @Override
    public INDArray backward(INDArray predicted, INDArray real) {
        if (singleClass) {
            return Util.unnormalize(this.softmaxPreds.sub(real));
        } else {
            return this.softmaxPreds.sub(real).div(real.shape()[1]);
        }
    }
}
