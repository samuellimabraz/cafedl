package br.cafedl.neuralnetwork.core.loss;

import br.cafedl.neuralnetwork.core.losses.BinaryCrossEntropy;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class BinaryCrossEntropyTest {

    @Test
    public void testForward() {
         INDArray yTrue = Nd4j.create(new double[][]{{1}, {0}, {1}, {1}});
         INDArray yPred = Nd4j.create(new double[][]{{0.4}, {0.5}, {0.8}, {0.2}});

         BinaryCrossEntropy loss = new BinaryCrossEntropy();

         System.out.println(yTrue);
         System.out.println(yPred);

         INDArray result = loss.forward(yPred, yTrue);
         System.out.println(result);

         INDArray back = loss.backward(yPred, yTrue);
         System.out.println(back);

    }
}
