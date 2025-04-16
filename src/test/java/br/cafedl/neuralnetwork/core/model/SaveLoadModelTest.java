package br.cafedl.neuralnetwork.core.model;

import br.cafedl.neuralnetwork.core.layers.Dense;
import br.cafedl.neuralnetwork.core.layers.Layer;
import br.cafedl.neuralnetwork.core.models.NeuralNetwork;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class SaveLoadModelTest {

    @Test
    public void testLoadLinearModel() throws Exception {
        // Carrega o modelo do arquivo
        String filePath = "src/main/resources/br/cafedl/neuralnetwork/examples/data/regression/linear_regression.zip";
        NeuralNetwork loadedModel = null;
        try {
           loadedModel = NeuralNetwork.loadModel(filePath);
        } catch (Exception e) {
            e.printStackTrace();
        }

        // Obtém as camadas do modelo
        List<Layer> layers = loadedModel.getLayers();

        // Verifica se o modelo tem a quantidade correta de camadas
        assertEquals(1, layers.size());

        // Verifica se a primeira camada é uma camada densa com a configuração correta
        assertTrue(layers.get(0) instanceof Dense);
        Dense denseLayer = (Dense) layers.get(0);
        assertEquals(1, denseLayer.getUnits());
        assertEquals("linear", denseLayer.getActivation().getClass().getSimpleName().toLowerCase());
        assertEquals("he", denseLayer.getKernelInitializer());

        System.out.println("Weights: " + denseLayer.getWeights());
        System.out.println("Bias: " + denseLayer.getBias());

//        Bias: [[172.3108],
//              [340.9742]]

    }

    @Test
    public void testLoadSineModel() {
        // Carrega o modelo do arquivo
        String filePath = "src/main/resources/br/cafedl/neuralnetwork/examples/data/regression/sine_function.zip";
        NeuralNetwork loadedModel = null;
        try {
            loadedModel = NeuralNetwork.loadModel(filePath);
        } catch (Exception e) {
            e.printStackTrace();
        }

        // Obtém as camadas do modelo
        List<Layer> layers = loadedModel.getLayers();

        // Verifica se o modelo tem a quantidade correta de camadas
        assertEquals(3, layers.size());

        // Verifica se a primeira camada é uma camada densa com a configuração correta
        assertTrue(layers.get(0) instanceof Dense);
        Dense denseLayer = (Dense) layers.get(0);
        assertEquals(32, denseLayer.getUnits());
        assertEquals("relu", denseLayer.getActivation().getClass().getSimpleName().toLowerCase());
        assertEquals("he", denseLayer.getKernelInitializer());

        INDArray params = denseLayer.getParams();
        INDArray weights = denseLayer.getWeights();
        INDArray bias = denseLayer.getBias();

//        Dense weights layer 0: [[   -0.2044,   -0.2708,   -0.0499,    0.6525,   -0.0006,   -0.0006,   -0.5470,    0.0006,    0.0006,   -0.0009,   -0.5157,   -0.0006,   -0.2431,   -0.4427,    1.2425,   -0.5424]]
//        Dense bias layer 0: [    0.6779,    0.3229,    0.0551,   -0.2932,   -0.2104,   -0.5298,   -0.2648,   -0.1456,   -0.3972,    0.1876,   -0.2498,   -0.3451,    0.2894,    0.5308,    0.5239,   -0.2626]
        System.out.println("Weights: " + weights);
        System.out.println("Bias: " + bias);
    }
}
