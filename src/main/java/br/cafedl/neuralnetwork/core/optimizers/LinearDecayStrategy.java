package br.cafedl.neuralnetwork.core.optimizers;

public class LinearDecayStrategy extends LearningRateDecayStrategy{
    public LinearDecayStrategy(double initialRate, double finalRate, int epochs) {
        super(initialRate, finalRate, epochs);
    }

    @Override
    public double calculateDecayPerEpoch(double initialRate, double finalRate, int epochs) {
        return (initialRate - finalRate) / (epochs);
    }

    @Override
    public double updateLearningRate() {
        this.learningRate -= decayPerEpoch;
        return this.learningRate;
    }
}
