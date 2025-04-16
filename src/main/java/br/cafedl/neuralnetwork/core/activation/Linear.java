package br.cafedl.neuralnetwork.core.activation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Linear implements IActivation {

        /**
        * Apply linear activation function to ndarray input
        * @param input (INDArray)
        * @return input (INDArray)
        */
        @Override
        public INDArray forward(INDArray input) {
            return input;
        }

        /**
        * Calculate derivative of linear activation function
        * @param input (INDArray)
        * @return 1 (INDArray)
        */
        @Override
        public INDArray backward(INDArray input) {
            return Nd4j.ones(input.shape());
        }

}
