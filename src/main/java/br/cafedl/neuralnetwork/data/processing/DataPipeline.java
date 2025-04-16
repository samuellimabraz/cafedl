package br.cafedl.neuralnetwork.data.processing;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.List;

public class DataPipeline extends DataProcessor {
    private List<DataProcessor> processors = new ArrayList<>();

    public DataPipeline(List<DataProcessor> processors) {
        this.processors = processors;
    }

    public DataPipeline() {
    }

    public void add(DataProcessor processor) {
        processors.add(processor);
    }

    public void fit(INDArray data) {
        for (DataProcessor processor : processors) {
            processor.fit(data);
        }
    }

    public INDArray transform(INDArray data) {
        for (DataProcessor processor : processors) {
            data = processor.transform(data);
        }
        return data;
    }

    @Override
    public INDArray fitTransform(INDArray data) {
        for (DataProcessor processor : processors) {
            processor.fit(data);
            data = processor.transform(data);
        }
        return data;
    }

    public INDArray inverseTransform(INDArray data) {
        for (int i = processors.size() - 1; i >= 0; i--) {
            data = processors.get(i).inverseTransform(data);
        }
        return data;
    }

    public List<DataProcessor> getProcessors() {
        return processors;
    }
}
