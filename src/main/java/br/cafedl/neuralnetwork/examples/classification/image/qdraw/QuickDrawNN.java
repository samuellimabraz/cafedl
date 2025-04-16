package br.cafedl.neuralnetwork.examples.classification.image.qdraw;

import br.cafedl.neuralnetwork.core.activation.Activation;
import br.cafedl.neuralnetwork.core.layers.Conv2D;
import br.cafedl.neuralnetwork.core.layers.Dense;
import br.cafedl.neuralnetwork.core.layers.Dropout;
import br.cafedl.neuralnetwork.core.layers.Flatten;
import br.cafedl.neuralnetwork.core.metrics.Accuracy;
import br.cafedl.neuralnetwork.core.metrics.F1Score;
import br.cafedl.neuralnetwork.core.metrics.Precision;
import br.cafedl.neuralnetwork.core.metrics.Recall;
import br.cafedl.neuralnetwork.core.models.ModelBuilder;
import br.cafedl.neuralnetwork.core.models.NeuralNetwork;
import br.cafedl.neuralnetwork.core.optimizers.*;
import br.cafedl.neuralnetwork.core.train.Trainer;
import br.cafedl.neuralnetwork.core.train.TrainerBuilder;
import br.cafedl.neuralnetwork.core.losses.SoftmaxCrossEntropy;
import br.cafedl.neuralnetwork.data.DataLoader;
import br.cafedl.neuralnetwork.data.Util;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.File;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

public class QuickDrawNN {

    public static List<String> CLASS_NAMES = Arrays.asList(
            "ladder",
            "bucket",
            "t-shirt",
            "tree",
            "dumbbell",
            "clock",
            "square",
            "triangle",
            "hourglass",
            "candle"
    );

    public static void countClassExamples(INDArray yTrain, INDArray yTest) {
        // Convert INDArrays to List<Integer>
        List<Integer> yTrainList = Arrays.stream(Nd4j.argMax(yTrain, 1).toIntVector()).boxed().collect(Collectors.toList());
        List<Integer> yTestList = Arrays.stream(Nd4j.argMax(yTest, 1).toIntVector()).boxed().collect(Collectors.toList());

        // Count and print the number of examples for each class in yTrain and yTest
        for (int i = 0; i < CLASS_NAMES.size(); i++) {
            int trainCount = Collections.frequency(yTrainList, i);
            int testCount = Collections.frequency(yTestList, i);
            System.out.println("Class " + CLASS_NAMES.get(i) + ": " + trainCount + " train examples, " + testCount + " test examples");
        }
    }

    public static void main(String[] args) throws Exception {
        //testLoadandPredict();
        train(2000, 500);
    }

    public static void train(int trainSize, int testSize) throws Exception {
        String root = "src/main/resources/br/deeplearning4java/neuralnetwork/examples/data/quickdraw";
        int numClasses = CLASS_NAMES.size();

        DataLoader dataLoader = new DataLoader(root + "/npy/train/x_train250.npy", root + "/npy/train/y_train250.npy", root + "/npy/test/x_test250.npy", root + "/npy/test/y_test250.npy");
        INDArray xTrain = dataLoader.getAllTrainImages().get(NDArrayIndex.interval(0, trainSize));
        INDArray yTrain = dataLoader.getAllTrainLabels().reshape(-1, 1).get(NDArrayIndex.interval(0, trainSize));
        INDArray xTest = dataLoader.getAllTestImages().get(NDArrayIndex.interval(0, testSize));
        INDArray yTest = dataLoader.getAllTestLabels().reshape(-1, 1).get(NDArrayIndex.interval(0, testSize));

        // Normalization
        xTrain = xTrain.divi(255);
        xTest = xTest.divi(255);
//        DataProcessor scaler = new MinMaxScaler();  // Normalização min-max
//        xTrain = scaler.fitTransform(xTrain);
//        xTest = scaler.transform(xTest);

        System.out.println("xTrain min - max: " + xTrain.minNumber() + " - " + xTrain.maxNumber());
        System.out.println("xTest min - max: " + xTest.minNumber() + " - " + xTest.maxNumber());

        // Reshape
        xTrain = xTrain.reshape(xTrain.rows(), 28, 28, 1);
        xTest = xTest.reshape(xTest.rows(), 28, 28, 1);

        // One-hot encoding
        yTrain = Util.oneHotEncode(yTrain, numClasses).reshape(-1, numClasses);
        yTest = Util.oneHotEncode(yTest, numClasses).reshape(-1, numClasses);

        System.out.println(xTrain.shapeInfoToString());
        System.out.println(yTrain.shapeInfoToString());
        System.out.println(xTest.shapeInfoToString());
        System.out.println(yTest.shapeInfoToString());

        // Count class examples
        countClassExamples(yTrain.dup(), yTest.dup());

        // Build the model
        NeuralNetwork model = new ModelBuilder()
                .add(new Conv2D(32, 2, Arrays.asList(2, 2), "valid", Activation.create("relu"), "he"))
                .add(new Conv2D(16, 1, Arrays.asList(1, 1), "valid", Activation.create("relu"), "he"))
                .add(new Flatten())
                .add(new Dense(178, Activation.create("relu"), "he"))
                .add(new Dropout(0.4))
                .add(new Dense(49, Activation.create("relu"), "he"))
                .add(new Dropout(0.3))
                .add(new Dense(numClasses,  Activation.create("linear"), "he"))
                .build();

        //NeuralNetwork model = NeuralNetwork.loadModel("src/main/resources/br/deeplearning4java/neuralnetwork/examples/data/quickdraw/model/quickdraw-cnn.zip");

        int epochs = 20;
        int batchSize = 64;

        LearningRateDecayStrategy lr = new ExponentialDecayStrategy(0.01, 0.0001, epochs);
        Optimizer optimizer = new RMSProp(lr);
        Trainer trainer = new TrainerBuilder(model, xTrain, yTrain, xTest, yTest, new SoftmaxCrossEntropy())
                .setOptimizer(optimizer)
                .setBatchSize(batchSize)
                .setEpochs(epochs)
                .setEvalEvery(2)
                .setEarlyStopping(true)
                .setPatience(4)
                .setMetric(new Accuracy())
                .build();
        trainer.fit();

       try {
           model.saveModel("src/main/resources/br/deeplearning4java/neuralnetwork/examples/data/quickdraw/model/quickdraw-cnn.zip");
           System.out.println("Model saved");
       } catch (Exception e) {
           e.printStackTrace();
       }

       model.setInference(true);

       // Test Predictions
        System.out.println("y_train[1] class: " + CLASS_NAMES.get(yTrain.getRow(1).argMax().getInt(0)));
        System.out.println("model.predict(x_train[1]): " + CLASS_NAMES.get(model.predict(xTrain.get(NDArrayIndex.point(1), NDArrayIndex.all()).reshape(1, 28,28,1)).argMax().getInt(0)));
        System.out.println("y_test[1] class: " + CLASS_NAMES.get(yTest.getRow(1).argMax().getInt(0)));
        System.out.println("model.predict(x_test[1]): " + CLASS_NAMES.get(model.predict(xTest.get(NDArrayIndex.point(1), NDArrayIndex.all()).reshape(1, 28,28,1)).argMax().getInt(0)));


        System.out.println("\nPredict vs True Train: ");
        printMetrics(model.predict(xTrain), yTrain);
        System.out.println("\nPredict vs True Test: ");
        printMetrics(model.predict(xTest), yTest);
    }

    public static void printMetrics(INDArray predict, INDArray y_test) {
        INDArray predictClass = predict.argMax(1);
        INDArray y_testClass = y_test.argMax(1);

        // Comparação entre os valores reais e os valores previstos
        INDArray comp = predictClass.eq(y_testClass).castTo(Nd4j.defaultFloatingPointType());
        System.out.println("True: " + comp.sumNumber().intValue());
        System.out.println("False: " + (y_testClass.length() - comp.sumNumber().intValue()));

        INDArray predictOneHot = Util.oneHotEncode(predictClass.reshape(predictClass.length(), 1), 10);

        // Metricas
        Accuracy accuracy = new Accuracy();
        double acc = accuracy.evaluate(y_testClass, predictClass);
        System.out.println("Accuracy: " + acc);
        Precision precision = new Precision();
        double prec = precision.evaluate(y_test, predictOneHot);
        System.out.println("Precision: " + prec);
        Recall recall = new Recall();
        double rec = recall.evaluate(y_test, predictOneHot);
        System.out.println("Recall: " + rec);
        F1Score f1Score = new F1Score();
        double f1 = f1Score.evaluate(y_test, predictOneHot);
        System.out.println("F1 Score: " + f1);

        INDArray cfm = Util.confusionMatrix(predictOneHot, y_test);
        System.out.println("Confusion Matrix: ");
        System.out.println(cfm.toStringFull());
    }

    public static void testLoadandPredict() throws Exception {
        String root = "src/main/resources/br/edu/unifei/ecot12/deeplearning4java/neuralnetwork/examples/data/quickdraw";

        NeuralNetwork model = NeuralNetwork.loadModel(root + "/model/quickdraw_model-cnn.zip");
        System.out.println("Model loaded");

        int numClasses = CLASS_NAMES.size();

        INDArray xTest = Nd4j.createFromNpyFile(new File(root + "/npy/test/x_test200.npy"));
        System.out.println(xTest.shapeInfoToString());
        INDArray yTest = Nd4j.createFromNpyFile(new File(root + "/npy/test/y_test200.npy")).reshape(-1, 1);
        System.out.println(yTest.shapeInfoToString());

        xTest = xTest.divi(255);
        // Reshape
        xTest = xTest.reshape(-1, 28, 28, 1);
        System.out.println("xTest shape: " + Arrays.toString(xTest.shape()));

        yTest = Util.oneHotEncode(yTest, 10).reshape(-1, 10);
        System.out.println("yTest min - max: " + yTest.minNumber() + " - " + yTest.maxNumber());

        System.out.println("y_test[10] class: " + CLASS_NAMES.get(yTest.getRow(10).argMax().getInt(0)));
        System.out.println("model.predict(x_test[10]): " + CLASS_NAMES.get(model.predict(xTest.get(NDArrayIndex.point(10), NDArrayIndex.all()).reshape(1, 28, 28, 1)).argMax().getInt(0)));

        printMetrics(model.predict(xTest), yTest);
    }
}
