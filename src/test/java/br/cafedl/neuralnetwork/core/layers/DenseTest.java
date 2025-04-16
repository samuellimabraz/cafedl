package br.cafedl.neuralnetwork.core.layers;

import br.cafedl.neuralnetwork.core.activation.Activation;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;

import static org.junit.jupiter.api.Assertions.*;

public class DenseTest {

    @Test
    public void testForward() {
        int units = 4;
        Dense dense = new Dense(units, Activation.create("linear"));
        INDArray input = Nd4j.create(new double[][]{{1, 2, 3, 4, 5}});
        INDArray output = dense.forward(input);


        System.out.println("Params:" + dense.getParams());
        System.out.println("Output:" + output);

        assertNotNull(output);
        assertEquals(units, output.length());
    }

    @Test
    public void testLinear() {
        Dense dense = new Dense(1, Activation.create("linear"));

        INDArray input = Nd4j.arange(1, 6).reshape(5, 1); // [[1], [2], [3], [4], [5]]
        INDArray expectedOutput = input.dup(); // a saída esperada é a mesma que a entrada

        System.out.println("Input:" + input);

        dense.setup(input);
        // Define os pesos e bias do modelo para valores conhecidos
        dense.setParams(Nd4j.vstack(Nd4j.ones(1, 1), Nd4j.zeros(1, 1)));
        System.out.println("Params:" + dense.getParams());

        // Testa o método forward
        INDArray output = dense.forward(input);
        System.out.println("Output:" + output);
        assertArrayEquals(expectedOutput.toDoubleVector(), output.toDoubleVector(), 1e-6);

        // Testa o método backward
        INDArray gradout = Nd4j.ones(5, 1); // o gradiente de saída é [1, 1, 1, 1, 1]
        INDArray expectedGradInput = gradout.dup(); // o gradiente de entrada esperado é o mesmo que o gradiente de saída
        INDArray gradInput = dense.backward(gradout);

        System.out.println("GradOut:" + gradout);
        System.out.println("GradInput:" + gradInput);
        System.out.println("ExpectedGradInput:" + expectedGradInput);

        assertArrayEquals(expectedGradInput.toDoubleVector(), gradInput.toDoubleVector(), 1e-6);
    }

    @Test
    public void testSaveAndLoad() throws Exception {
        // Cria uma camada densa
        Dense originalLayer = new Dense(10, Activation.create("relu"), "standard");
        originalLayer.forward(Nd4j.ones(DataType.DOUBLE, 1, 5));

        // Salva a camada em um arquivo
        String filePath = "src/test/java/br/cafedl/neuralnetwork/core/layers/testLayer.bin";
        DataOutputStream dos = new DataOutputStream(new FileOutputStream(filePath));
        originalLayer.save(dos);
        dos.close();

        // Carrega a camada do arquivo
        DataInputStream dis = new DataInputStream(new FileInputStream(filePath));
        Dense loadedLayer = (Dense) LayerLoader.load(dis);

        System.out.println("Original Layer:");
        System.out.println(originalLayer.toString());
        System.out.println("Loaded Layer:");
        System.out.println(loadedLayer.toString());

        // Verifica se a camada carregada é igual à original
        assertEquals(originalLayer.getUnits(), loadedLayer.getUnits());
        assertEquals(originalLayer.getKernelInitializer(), loadedLayer.getKernelInitializer());
        assertTrue(originalLayer.getParams().equalsWithEps(loadedLayer.getParams(), 1e-5));

        DataInputStream dis2 = new DataInputStream(new FileInputStream(filePath));
        Dense loadedLayer2 = (Dense) LayerLoader.load(dis2);

        System.out.println("Loaded Layer 2:");
        System.out.println(loadedLayer2.toString());

        // Verifica o endereco de memoria
        assertNotEquals(loadedLayer, loadedLayer2);

        // Apaga o arquivo
        new File(filePath).delete();
    }
}
