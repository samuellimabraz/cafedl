package br.cafedl.neuralnetwork.examples.classification.image.mnist;

import br.cafedl.neuralnetwork.core.activation.Activation;
import br.cafedl.neuralnetwork.core.layers.Dense;
import br.cafedl.neuralnetwork.core.layers.Dropout;
import br.cafedl.neuralnetwork.core.layers.Flatten;
import br.cafedl.neuralnetwork.core.metrics.Accuracy;
import br.cafedl.neuralnetwork.core.metrics.F1Score;
import br.cafedl.neuralnetwork.core.models.ModelBuilder;
import br.cafedl.neuralnetwork.core.models.NeuralNetwork;
import br.cafedl.neuralnetwork.core.optimizers.ExponentialDecayStrategy;
import br.cafedl.neuralnetwork.core.optimizers.LearningRateDecayStrategy;
import br.cafedl.neuralnetwork.core.optimizers.Optimizer;
import br.cafedl.neuralnetwork.core.optimizers.RMSProp;
import br.cafedl.neuralnetwork.core.train.Trainer;
import br.cafedl.neuralnetwork.core.train.TrainerBuilder;
import br.cafedl.neuralnetwork.data.DataLoader;
import br.cafedl.neuralnetwork.data.Util;
import br.cafedl.neuralnetwork.data.processing.DataProcessor;
import br.cafedl.neuralnetwork.data.processing.StandardScaler;
import br.cafedl.neuralnetwork.core.losses.SoftmaxCrossEntropy;
import org.nd4j.linalg.api.ndarray.INDArray;

import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.IOException;

public class MnistNN {

    public static void train(int trainSize, int testSize) throws IOException {
        DataLoader mnistDataLoader = new DataLoader("src/main/resources/br/deeplearning4java/neuralnetwork/examples/data/mnist/train/mnist_train.bin", "src/main/resources/br/deeplearning4java/neuralnetwork/examples/data/mnist/test/mnist_test.bin");

        // Images
        INDArray x_train = mnistDataLoader.getAllTrainImages();
        INDArray x_test = mnistDataLoader.getAllTestImages();
        INDArray y_train = mnistDataLoader.getAllTrainLabels();
        INDArray y_test = mnistDataLoader.getAllTestLabels();

        // Pega uma fatia dos dados
        x_train = x_train.get(NDArrayIndex.interval(0, trainSize));
        y_train = y_train.get(NDArrayIndex.interval(0, trainSize));
        x_test = x_test.get(NDArrayIndex.interval(0, testSize));
        y_test = y_test.get(NDArrayIndex.interval(0, testSize));

        // Min-Max Scaling
        x_train = x_train.divi(255); // Normalization (0 - 1)
        x_test = x_test.divi(255); // Normalization (0 - 1)
//        DataProcessor scaler = new MinMaxScaler();
//        x_train = scaler.fitTransform(x_train);
//        x_test = scaler.transform(x_test);

        x_train = x_train.reshape(x_train.rows(), 28, 28, 1);
        x_test = x_test.reshape(x_test.rows(), 28, 28, 1);

        // One hot encoding
        y_train = Util.oneHotEncode(y_train, 10);
        y_test = Util.oneHotEncode(y_test, 10);

        // Network
        NeuralNetwork model = new ModelBuilder()
                .add(new Flatten())
                .add(new Dense(89, Activation.create("relu"), "he"))
                .add(new Dropout(0.4))
                .add(new Dense(32, Activation.create("relu"), "he"))
                .add(new Dropout(0.4))
                .add(new Dense(10, Activation.create("linear"), "he"))
                .build();

        // Training
        LearningRateDecayStrategy lr = new ExponentialDecayStrategy(0.01, 0.001, 30);
        Optimizer optimizer = new RMSProp(lr);
        Trainer trainer = new TrainerBuilder(model, x_train, y_train, x_test, y_test, new SoftmaxCrossEntropy())
                .setOptimizer(optimizer)
                .setEpochs(30)
                .setBatchSize(128)
                .setEvalEvery(2)
                .setEarlyStopping(true)
                .setPatience(4)
                .setMetric(new Accuracy())
                .build();

        trainer.printDataInfo();
        trainer.fit();

        // Evaluation
        evaluate(model, x_test, y_test);

        try {
                model.saveModel("src/main/resources/br/deeplearning4java/neuralnetwork/examples/data/mnist/model/mnist_model.bin");
        } catch (IOException e) {
                e.printStackTrace();
        }

    }

    public static void evaluate(NeuralNetwork model, INDArray testX, INDArray testY) {
        // Evaluation
        model.setInference(true);
        INDArray predictions = model.predict(testX);

        INDArray predictionsClasses = predictions.argMax(1);
        INDArray predictionsClassesOneHot = Util.oneHotEncode(predictionsClasses.reshape(predictionsClasses.length(), 1), 10);
        INDArray testLabelsClasses = testY.argMax(1);

        Accuracy accuracy = new Accuracy();
        double acc = accuracy.evaluate(testLabelsClasses, predictionsClasses);
        System.out.println("Accuracy: " + acc);
        F1Score f1Score = new F1Score();
        double f1 = f1Score.evaluate(testY, predictionsClassesOneHot);
        System.out.println("F1 Score: " + f1);

        INDArray cmf = Util.confusionMatrix(predictionsClassesOneHot, testY);
        System.out.println("Confusion Matrix: ");
        System.out.println(cmf.toStringFull());
    }

    public static void testModel(int testSize) throws Exception {
        DataLoader mnistDataLoader = new DataLoader("src/main/resources/br/edu/unifei/ecot12/deeplearning4java/neuralnetwork/examples/data/mnist/train/mnist_train.bin", "src/main/resources/br/edu/unifei/ecot12/deeplearning4java/neuralnetwork/examples/data/mnist/test/mnist_test.bin");

        // Images
        INDArray x_train = mnistDataLoader.getAllTrainImages();
        INDArray x_test = mnistDataLoader.getAllTestImages();
        INDArray y_test = mnistDataLoader.getAllTestLabels();

        // Pega uma fatia dos dados
        x_test = x_test.get(NDArrayIndex.interval(0, testSize));
        y_test = y_test.get(NDArrayIndex.interval(0, testSize));

        // Scaler
        DataProcessor scaler = new StandardScaler();
        x_train = scaler.fitTransform(x_train);
        x_test = scaler.transform(x_test);

        //x_test = x_test.reshape(x_test.rows(), 28, 28, 1);

        // One hot encoding
        y_test = Util.oneHotEncode(y_test, 10);

        NeuralNetwork model = NeuralNetwork.loadModel("src/main/resources/br/edu/unifei/ecot12/deeplearning4java/neuralnetwork/examples/data/mnist/model/mnist_model.bin");

        // Evaluation
        evaluate(model, x_test, y_test);
    }
    public static void main(String[] args) throws Exception {
        train(1920, 480);
        //testModel(60);
    }
}
