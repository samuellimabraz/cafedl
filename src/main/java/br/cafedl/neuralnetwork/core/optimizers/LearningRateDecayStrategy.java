package br.cafedl.neuralnetwork.core.optimizers;

public abstract class LearningRateDecayStrategy {
    double decayPerEpoch, learningRate;

    public LearningRateDecayStrategy(double initialRate, double finalRate, int epochs) {
        this.learningRate = initialRate;
        decayPerEpoch = this.calculateDecayPerEpoch(initialRate, finalRate, epochs);
    }

    protected abstract double calculateDecayPerEpoch(double initialRate, double finalRate, int epochs);

    public abstract double updateLearningRate();
}
