package br.cafedl.neuralnetwork.core.train;

import br.cafedl.neuralnetwork.core.losses.ILossFunction;
import br.cafedl.neuralnetwork.core.metrics.IMetric;
import br.cafedl.neuralnetwork.data.Util;
import br.cafedl.neuralnetwork.core.models.NeuralNetwork;
import br.cafedl.neuralnetwork.core.optimizers.Optimizer;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Random;


public class Trainer {

    private final NeuralNetwork model;

    private final Optimizer optimizer;

    private final ILossFunction lossFunction;
    private IMetric metric=null;
    
    private INDArray trainInputs;
    private INDArray trainTargets;
    private INDArray testInputs;
    private INDArray testTargets;

    private INDArray[] batch = new INDArray[2];

    private final int epochs;
    private final int batchSize;
    private int currentIndex;
    private final int patience;

    private int evalEvery = 10;
    private final boolean earlyStopping, verbose;

    private double bestLoss = Double.MAX_VALUE;

    private int wait = 0;

    private final double threshold = Double.MIN_VALUE;

    private double trainLoss, valLoss, trainMetricValue, valMetricValue;

    public Trainer(TrainerBuilder builder) {
        this.model = builder.model;
        this.model.setInference(false);
        this.optimizer = builder.optimizer;
        this.lossFunction = builder.lossFunction;
        this.metric = builder.metric;
        this.patience = builder.patience;
        this.earlyStopping = builder.earlyStopping;
        this.verbose = builder.verbose;
        this.epochs = builder.epochs;
        this.evalEvery = builder.evalEvery;
        this.batchSize = builder.batchSize;
        double trainRatio = builder.trainRatio;
        this.trainInputs = builder.trainInputs;
        this.trainTargets = builder.trainTargets;
        if (builder.testInputs != null && builder.testTargets != null) {
            this.testInputs = builder.testInputs;
            this.testTargets = builder.testTargets;
        } else {
            this.splitData(builder.trainInputs, builder.trainTargets, trainRatio);
        }
    }

    public void fit() {
        System.out.println(" -- Training...");
        System.out.println("Epochs: " + epochs);
        System.out.println("Batch size: " + batchSize);
        INDArray Trainpredictions = null, Testpredictions = null, Batchpredictions = null, gradLoss = null;
        INDArray indices = Nd4j.arange(trainInputs.shape()[0]);
        Random rnd = new Random(42);

        Runtime runtime = Runtime.getRuntime();
        long usedMemoryBefore = runtime.totalMemory() - runtime.freeMemory();
        System.out.println("Used memory before training: " + usedMemoryBefore);

        long startTime = System.currentTimeMillis();
        double batchLoss;


        for (int epoch = 0; epoch < epochs; epoch++) {
            currentIndex = 0;

            // Shuffle the training data at the start of each epoch
            // Create an array of indices and shuffle it
            Nd4j.shuffle(indices, rnd,0);

            // Use the shuffled indices to reorder trainInputs and trainTargets
            trainInputs = trainInputs.get(indices);
            trainTargets = trainTargets.get(indices);

            System.out.println("\nEpoch: " + (epoch + 1));
            while (hasNextBatch()) {
                this.getNextBatch();

                // Forward pass
                Batchpredictions = model.predict(batch[0].dup());

                batchLoss = lossFunction.forward(Batchpredictions.dup(), batch[1].dup()).getDouble(0);

                if (verbose) {
                    Util.printProgressBar(currentIndex, (int) trainInputs.size(0));
                    System.out.print(" " + currentIndex + "/" + trainInputs.size(0));
                    System.out.print(" | Batch Loss: " + batchLoss);
                }

                // Backward pass
                gradLoss = lossFunction.backward(Batchpredictions, batch[1]);

                model.backPropagation(gradLoss);

                optimizer.update();
            }
            optimizer.updateEpoch();

            if ((epoch+1) % evalEvery == 0) {
                this.model.setInference(true);
                Testpredictions = model.predict(testInputs.dup());
                Trainpredictions = model.predict(trainInputs.dup());
                valLoss = lossFunction.forward(Testpredictions.dup(), testTargets.dup()).getDouble(0);
                trainLoss = lossFunction.forward(Trainpredictions.dup(), trainTargets.dup()).getDouble(0);
                this.model.setInference(false);

                if (verbose) {
                    System.out.print(" | Loss: " + trainLoss +  " | Test Loss: " + valLoss);
                    if (metric != null) {
                        String metricName = metric.getClass().getSimpleName();
                        valMetricValue = metric.evaluate(testTargets.dup().argMax(1), Testpredictions.dup().argMax(1));
                        trainMetricValue = metric.evaluate(trainTargets.dup().argMax(1), Trainpredictions.dup().argMax(1));
                        System.out.print(" | " + metricName + ": " + trainMetricValue + " | Test " + metricName + ": " + valMetricValue);
                    }
                }
                if (earlyStopping) {
                    if (earlyStopping()) {
                        System.out.println("\nEarly stopping at epoch: " + (epoch+1) + " Loss: " + valLoss + " | Best Loss: " + bestLoss);
                        epoch = epochs;
                    }
                }
            }
        }

        long endTime = System.currentTimeMillis();
        System.out.println("\nTraining time: " + (endTime - startTime) / 1000.0 + " seconds");

        long usedMemoryAfter = runtime.totalMemory() - runtime.freeMemory();
        System.out.println("Used memory after training: " + usedMemoryAfter);
        System.out.println("Memory used for training: " + (usedMemoryAfter - usedMemoryBefore));
    }

    public void evaluate() {
        valLoss = lossFunction.forward(model.predict(testInputs), testTargets).getDouble(0) + threshold;
        System.out.println("Test Loss: " + valLoss);
    }

    private boolean earlyStopping() {
        if (valLoss < (bestLoss - threshold)) {
            wait = 0;
            bestLoss = valLoss;
        } else if ((valLoss > (bestLoss + threshold)) || (valLoss == bestLoss) || (valLoss <= threshold)) {
            wait++;
        }

        return (valLoss <= threshold) || (wait >= patience);
    }

    private boolean hasNextBatch() {
        return currentIndex < trainInputs.size(0);
    }

    private void getNextBatch() {
        int end = (int) Math.min(currentIndex + batchSize, trainInputs.size(0));
        batch[0] = trainInputs.get(NDArrayIndex.interval(currentIndex, end));
        batch[1] = trainTargets.get(NDArrayIndex.interval(currentIndex, end));

        if (batch[0].size(0) < batchSize) {
            int remaining = (int) (batchSize - batch[0].size(0));
            INDArray[] newBatch = new INDArray[2];
            newBatch[0] = trainInputs.get(NDArrayIndex.interval(0, remaining));
            newBatch[1] = trainTargets.get(NDArrayIndex.interval(0, remaining));
            batch[0] = Nd4j.concat(0, batch[0], newBatch[0]);
            batch[1] = Nd4j.concat(0, batch[1], newBatch[1]);
        }

        currentIndex = end;
    }

    public void splitData(INDArray inputs, INDArray targets, double trainRatio) {
        INDArray[][] trainTestSplit = Util.trainTestSplit(inputs, targets, trainRatio);
        this.trainInputs = trainTestSplit[0][0];
        this.trainTargets = trainTestSplit[0][1];
        this.testInputs = trainTestSplit[1][0];
        this.testTargets = trainTestSplit[1][1];
    }

    public void printDataInfo() {
        System.out.println("Train inputs shape: " + trainInputs.shapeInfoToString());
        System.out.println("Train targets shape: " + trainTargets.shapeInfoToString());
        System.out.println("Test inputs shape: " + testInputs.shapeInfoToString());
        System.out.println("Test targets shape: " + testTargets.shapeInfoToString());
    }

    public INDArray getTrainInputs() {
        return this.trainInputs;
    }

    public INDArray getTrainTargets() {
        return this.trainTargets;
    }

    public INDArray getTestInputs() {
        return this.testInputs;
    }

    public INDArray getTestTargets() {
        return this.testTargets;
    }

}
