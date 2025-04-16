package br.cafedl.neuralnetwork.core.layers;

import br.cafedl.neuralnetwork.core.activation.Activation;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class Conv2DTest {

    @Test
    public void testForward() {
        // Create an instance of Conv2D
        int filters = 4;
        int kernelSize = 2;
        List<Integer> strides = Arrays.asList(1, 1);
        String padding = "same";
        Conv2D conv2D = new Conv2D(filters, kernelSize, strides, padding, Activation.create("relu"), "zero");

        // Create some input data
        INDArray inputs = Nd4j.rand(DataType.DOUBLE,1, 5, 5, 3); // 1 image, 5x5 size, 3 channels

        System.out.println("Inputs:" + inputs);

        // Call the forward method
        INDArray output = conv2D.forward(inputs);

        System.out.println("Params shape:" + conv2D.getParams().shapeInfoToString());
        System.out.println("Params:" + conv2D.getParams());

        System.out.println("Weights shape:" + conv2D.getWeights().shapeInfoToString());
        System.out.println("Weights:" + conv2D.getWeights());
        System.out.println("Biases shape:" + conv2D.getBiases().shapeInfoToString());
        System.out.println("Biases:" + conv2D.getBiases());

        // Check the shape of the output
        long[] shape = output.shape();

        System.out.println("Output:" + output);
        System.out.println("Output Shape:" + output.shapeInfoToString());

        assertNotNull(shape);
        assertEquals(4, shape.length);
        assertEquals(1, shape[0]); // should match the batch size of the input
        assertEquals(6, shape[1]); // should be (input height - kernel size + 2 * pad) / stride + 1
        assertEquals(6, shape[2]); // should be (input width - kernel size + 2 * pad) / stride + 1
        assertEquals(filters, shape[3]); // should match the number of filters
    }

    @Test
    public void testBackward() {
        // Create an instance of Conv2D
        int filters = 4;
        int kernelSize = 2;
        List<Integer> strides = Arrays.asList(1, 1);
        String padding = "same";
        Conv2D conv2D = new Conv2D(filters, kernelSize, strides, padding, Activation.create("relu"), "xavier");

        // Create some input data
        INDArray inputs = Nd4j.rand(1, 5, 5, 3); // 1 image, 5x5 size, 3 channels

        // Call the forward method
        INDArray output = conv2D.forward(inputs);

        System.out.println("Output shape:" + output.shapeInfoToString());

        // Create some gradient data
        INDArray gradout = Nd4j.rand(DataType.DOUBLE, output.shape());

        // Call the backward method
        INDArray gradInput = conv2D.backward(gradout);

        // Check the shape of the gradient input
        long[] shape = gradInput.shape();

        System.out.println("GradInput shape:" + gradInput.shapeInfoToString());

        assertNotNull(shape);
        assertEquals(4, shape.length);
        assertEquals(1, shape[0]); // should match the batch size of the input
        assertEquals(5, shape[1]); // should match the height of the input
        assertEquals(5, shape[2]); // should match the width of the input
        assertEquals(3, shape[3]); // should match the number of channels of the input
    }

    @Test
    public void testSaveAndLoad() throws Exception {
        Conv2D originalLayer = new Conv2D(4, 2, Arrays.asList(2, 2), "same", Activation.create("relu"), "xavier");
        // Create some input data
        INDArray inputs = Nd4j.rand(1, 5, 5, 3); // 1 image, 5x5 size, 3 channels

        originalLayer.forward(inputs);

        String filePath = "src/test/java/br/cafedl/neuralnetwork/core/layers/testConv2DLayer.bin";
        DataOutputStream dos = new DataOutputStream(new FileOutputStream(filePath));
        originalLayer.save(dos);
        dos.close();

        // Carrega a camada do arquivo
        DataInputStream dis = new DataInputStream(new FileInputStream(filePath));
        Conv2D loadedLayer = (Conv2D) LayerLoader.load(dis);

        System.out.println("Original Layer:");
        System.out.println(originalLayer.toString());
        System.out.println("Loaded Layer:");
        System.out.println(loadedLayer.toString());

        // Verifica se a camada carregada é igual à original
        assertEquals(originalLayer.toString(), loadedLayer.toString());

        DataInputStream dis2 = new DataInputStream(new FileInputStream(filePath));
        Conv2D loadedLayer2 = (Conv2D) LayerLoader.load(dis2);

        System.out.println("Loaded Layer 2:");
        System.out.println(loadedLayer2.toString());

        // Verifica o endereco de memoria
        assertNotEquals(loadedLayer, loadedLayer2);

        // Apaga o arquivo
        new File(filePath).delete();
    }
}
