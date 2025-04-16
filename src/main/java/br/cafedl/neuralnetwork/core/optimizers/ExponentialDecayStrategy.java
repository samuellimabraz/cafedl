package br.cafedl.neuralnetwork.core.optimizers;

public class ExponentialDecayStrategy extends LearningRateDecayStrategy {
    public ExponentialDecayStrategy(double initialRate, double finalRate, int epochs) {
        super(initialRate, finalRate, epochs);
    }

    @Override
    public double calculateDecayPerEpoch(double initialRate, double finalRate, int epochs) {
        return Math.pow(finalRate / initialRate, 1.0 / (epochs));
    }

    @Override
    public double updateLearningRate() {
        this.learningRate *= decayPerEpoch;
        return this.learningRate;
    }
}
