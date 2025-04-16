package br.cafedl.neuralnetwork.core.model;

import br.cafedl.neuralnetwork.core.activation.Activation;
import br.cafedl.neuralnetwork.core.activation.Linear;
import br.cafedl.neuralnetwork.core.activation.ReLU;
import br.cafedl.neuralnetwork.core.activation.Softmax;
import br.cafedl.neuralnetwork.core.layers.Dense;
import br.cafedl.neuralnetwork.core.models.ModelBuilder;
import br.cafedl.neuralnetwork.core.models.NeuralNetwork;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class NeuralNetworkTest {

    @Test
    public void testSimpleBuilder() {
        NeuralNetwork model = new ModelBuilder().add(new Dense(1, Activation.create("linear"))).build();

        assertTrue(model.getLayers().get(0) instanceof Dense);
        assertTrue(((Dense) model.getLayers().get(0)).getActivation() instanceof Linear);

    }

    @Test
    public void testComplexBuilder() {
        NeuralNetwork model = new ModelBuilder()
                .add(new Dense(28*28, Activation.create("relu")))
                .add(new Dense(64, Activation.create("linear")))
                .add(new Dense(10, Activation.create("softmax")))
                .build();

        assertTrue(model.getLayers().get(0) instanceof Dense);
        assertTrue(((Dense) model.getLayers().get(0)).getActivation() instanceof ReLU);
        assertEquals(28 * 28, ((Dense) model.getLayers().get(0)).getUnits());

        assertTrue(model.getLayers().get(1) instanceof Dense);
        assertTrue(((Dense) model.getLayers().get(1)).getActivation() instanceof Linear);
        assertEquals(64, ((Dense) model.getLayers().get(1)).getUnits());

        assertTrue(model.getLayers().get(2) instanceof Dense);
        assertTrue(((Dense) model.getLayers().get(2)).getActivation() instanceof Softmax);
        assertEquals(10, ((Dense) model.getLayers().get(2)).getUnits());

    }

    @Test
    public void testPredict() {
        NeuralNetwork model = new ModelBuilder()
                .add(new Dense(28*28, Activation.create("relu")))
                .add(new Dense(64, Activation.create("relu")))
                .add(new Dense(10, Activation.create("softmax")))
                .build();

        INDArray inputs = Nd4j.rand(DataType.DOUBLE,1, 28 * 28);
        INDArray predictions = model.predict(inputs);

        assertEquals(1, predictions.size(0));
        assertEquals(10, predictions.size(1));
    }
}
