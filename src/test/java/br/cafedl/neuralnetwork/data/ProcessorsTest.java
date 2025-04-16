package br.cafedl.neuralnetwork.data;

import br.cafedl.neuralnetwork.data.processing.DataPipeline;
import br.cafedl.neuralnetwork.data.processing.DataProcessor;
import br.cafedl.neuralnetwork.data.processing.MinMaxScaler;
import br.cafedl.neuralnetwork.data.processing.StandardScaler;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class ProcessorsTest {
    @Test
    public void testMinMax() {
        MinMaxScaler scaler = new MinMaxScaler(0, 1);
        INDArray data = Nd4j.create(new float[]{1, 2, 3, 4, 5}, new int[]{5, 1});
        scaler.fit(data);
        INDArray transformedData = scaler.transform(data);

        System.out.println("data: " + data);
        System.out.println("transformedData: " + transformedData);

        assertEquals(0, transformedData.getDouble(0, 0), "Expected 0");
        assertEquals(1, transformedData.getDouble(4, 0), "Expected 1");
    }

    @Test
    public void testStandardScaler() {
        StandardScaler scaler = new StandardScaler();
        INDArray data = Nd4j.create(new float[]{1, 2, 3, 4, 5}, new int[]{5, 1});
        scaler.fit(data);
        INDArray transformedData = scaler.transform(data);

        System.out.println("data: " + data);
        System.out.println("transformedData: " + transformedData);

        assertEquals(0, transformedData.meanNumber().doubleValue(), 1e-6, "Expected mean 0");
        assertEquals(1, transformedData.stdNumber().doubleValue(), 1e-6, "Expected std 1");
    }

    @Test
    public void testPipeline() {
        // Test MinMaxScaler
        List<DataProcessor> processors = List.of(new MinMaxScaler(0, 1));
        DataPipeline pipeline = new DataPipeline(processors);
        INDArray data = Nd4j.create(new float[]{1, 2, 3, 4, 5}, new int[]{5, 1});
        pipeline.fit(data);
        INDArray transformedData = pipeline.transform(data);
        assertEquals(0, transformedData.minNumber().doubleValue(), 1e-6, "Expected min 0");
        assertEquals(1, transformedData.maxNumber().doubleValue(), 1e-6, "Expected max 1");

        // Test StandardScaler
        processors = List.of(new StandardScaler());
        pipeline = new DataPipeline(processors);
        pipeline.fit(data);
        transformedData = pipeline.transform(data);
        assertEquals(0, transformedData.meanNumber().doubleValue(), 1e-6, "Expected mean 0");
        assertEquals(1, transformedData.stdNumber().doubleValue(), 1e-6, "Expected std 1");
    }
}
