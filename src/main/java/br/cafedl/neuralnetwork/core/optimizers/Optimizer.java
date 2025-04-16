package br.cafedl.neuralnetwork.core.optimizers;

import br.cafedl.neuralnetwork.core.layers.TrainableLayer;
import br.cafedl.neuralnetwork.core.models.NeuralNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public abstract class Optimizer {
    protected NeuralNetwork neuralNetwork;
    protected LearningRateDecayStrategy learningRateDecayStrategy;
    protected double learningRate;
    protected Map<TrainableLayer, List<INDArray>> auxParams = new HashMap<>();
    protected List<TrainableLayer> trainableLayers;

    private boolean initialized = false;

    public Optimizer() {}

    protected Optimizer(double learningRate) {
        this.learningRate = learningRate;
        this.learningRateDecayStrategy = null;
    }

    public Optimizer(LearningRateDecayStrategy learningRateDecayStrategy) {
        this.learningRateDecayStrategy = learningRateDecayStrategy;
        this.learningRate = learningRateDecayStrategy.learningRate;
    }

    public Optimizer(NeuralNetwork neuralNetwork) {
        this.neuralNetwork = neuralNetwork;
    }

    public void setNeuralNetwork(NeuralNetwork neuralNetwork) {
        this.neuralNetwork = neuralNetwork;
        this.trainableLayers = neuralNetwork.getTrainableLayers();
    }

    protected void init() {
        for (TrainableLayer layer : trainableLayers) {
            auxParams.put(layer, createAuxParams(layer.getParams()));
        }
    }

    public void update() {
        if (!initialized) {
            init();
            initialized = true;
        }
        for (TrainableLayer layer : trainableLayers) {
            updateRule(layer.getParams(), layer.getGrads(), auxParams.get(layer));
        }
    }

    public void updateEpoch() {
        if (learningRateDecayStrategy != null) {
            learningRate = learningRateDecayStrategy.updateLearningRate();
        }
    }

    protected abstract List<INDArray> createAuxParams(INDArray params);

    protected abstract void updateRule(INDArray params, INDArray grads, List<INDArray> auxParams);
}