package br.cafedl.neuralnetwork.core.train;

import br.cafedl.neuralnetwork.core.losses.ILossFunction;
import br.cafedl.neuralnetwork.core.metrics.IMetric;
import br.cafedl.neuralnetwork.core.models.NeuralNetwork;
import br.cafedl.neuralnetwork.core.optimizers.Adam;
import br.cafedl.neuralnetwork.core.optimizers.Optimizer;
import org.nd4j.linalg.api.ndarray.INDArray;

public class TrainerBuilder {
    public final NeuralNetwork model;
    public ILossFunction lossFunction;
    public IMetric metric = null;

    public Optimizer optimizer = new Adam(0.01);

    public INDArray trainInputs;
    public INDArray trainTargets;
    public INDArray testInputs = null;
    public INDArray testTargets = null;
    public int epochs = 100;
    public int batchSize = 32;
    public boolean earlyStopping = false, verbose = true;
    public int patience = 20;
    public int evalEvery = 10;
    public double trainRatio = 0.8;

    public TrainerBuilder(NeuralNetwork model, INDArray trainInputs, INDArray trainTargets, ILossFunction lossFunction) {
        this.model = model;
        this.trainInputs = trainInputs;
        this.trainTargets = trainTargets;
        this.lossFunction = lossFunction;
    }

    public TrainerBuilder(NeuralNetwork model, INDArray trainInputs, INDArray trainTargets, INDArray testInputs, INDArray testTargets, ILossFunction lossFunction) {
        this.model = model;
        this.trainInputs = trainInputs;
        this.trainTargets = trainTargets;
        this.testInputs = testInputs;
        this.testTargets = testTargets;
        this.lossFunction = lossFunction;
        this.trainRatio = 1.0;
    }

    public TrainerBuilder setOptimizer(Optimizer optimizer) {
        if (optimizer == null) throw new IllegalArgumentException("Optimizer cannot be null");
        this.optimizer = optimizer;
        return this;
    }

    public TrainerBuilder setEpochs(int epochs) {
        if (epochs <= 0) throw new IllegalArgumentException("Epochs must be greater than 0");
        this.epochs = epochs;
        return this;
    }

    public TrainerBuilder setBatchSize(int batchSize) {
        if (batchSize <= 0) throw new IllegalArgumentException("Batch size must be greater than 0");
        if (batchSize > trainInputs.size(0)) throw new IllegalArgumentException("Batch size must be less than the number of inputs");
        this.batchSize = batchSize;
        return this;
    }

    public TrainerBuilder setEarlyStopping(boolean earlyStopping) {
        this.earlyStopping = earlyStopping;
        return this;
    }

    public TrainerBuilder setPatience(int patience) {
        if (patience < 0) throw new IllegalArgumentException("Patience must be greater than 0");
        this.patience = patience;
        return this;
    }

    public TrainerBuilder setTrainRatio(double trainRatio) {
        if (trainRatio < 0.0 || trainRatio > 1.0) throw new IllegalArgumentException("Train ratio must be between 0 and 1");
        this.trainRatio = trainRatio;
        return this;
    }

    public TrainerBuilder setEvalEvery(int evalEvery) {
        if (evalEvery < 0 || evalEvery > epochs) throw new IllegalArgumentException("Eval every must be greater than 0");
        this.evalEvery = evalEvery;
        return this;
    }

    public TrainerBuilder setVerbose(boolean verbose) {
        this.verbose = verbose;
        return this;
    }

    public TrainerBuilder setMetric(IMetric metric) {
        this.metric = metric;
        return this;
    }

    public Trainer build() {
        optimizer.setNeuralNetwork(model);
        return new Trainer(this);
    }
}
