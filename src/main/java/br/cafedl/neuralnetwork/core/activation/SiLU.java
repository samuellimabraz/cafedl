package br.cafedl.neuralnetwork.core.activation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * SiLU (Sigmoid Linear Unit) activation function
 */
public class SiLU implements IActivation {
        private final Sigmoid sigmoid = new Sigmoid();

        /**
         * Aplly SiLU activation function to ndarray input
         * @param input
         * @return input * sigmoid(input) = input * 1 / (1 + exp(-input))
         */
        @Override
        public INDArray forward(INDArray input) {
            return input.mul(sigmoid.forward(input));
        }

        /**
         * Calculate derivative of SiLU activation function
         * @param input
         * @return sigmoid(input) + input * sigmoid(input) * (1 - sigmoid(input))
         */
        @Override
        public INDArray backward(INDArray input) {
//            INDArray sigmoid = this.sigmoid.forward(input);
//            return sigmoid.add(input.mul(sigmoid.rsub(1)));
            // Derivative of SiLU(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
            INDArray sigmoid = Transforms.sigmoid(input, true);
            return sigmoid.add(input.mul(sigmoid).mul(Nd4j.onesLike(input).sub(sigmoid)));
        }
}
