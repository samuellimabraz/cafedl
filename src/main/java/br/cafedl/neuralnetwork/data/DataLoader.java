package br.cafedl.neuralnetwork.data;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class DataLoader {
    private INDArray trainData;
    private INDArray testData;

    public DataLoader(String trainDataPath, String testDataPath) throws IOException {
        File trainDataFile = new File(trainDataPath);
        File testDataFile = new File(testDataPath);

        if (trainDataFile.exists() && testDataFile.exists()) {
            if (trainDataPath.endsWith(".bin") && testDataPath.endsWith(".bin")) {
                trainData = Nd4j.readBinary(trainDataFile);
                testData = Nd4j.readBinary(testDataFile);
            } else if (trainDataPath.endsWith(".npy") && testDataPath.endsWith(".npy")) {
                trainData = Nd4j.createFromNpyFile(trainDataFile);
                testData = Nd4j.createFromNpyFile(testDataFile);
            }
            else if (trainDataPath.endsWith(".csv") && testDataPath.endsWith(".csv")) {
                trainData = loadCsv(trainDataPath);
                testData = loadCsv(testDataPath);
            }
        } else {
            throw new IOException("Data files not found");
        }

        // Flatten the data
        trainData = trainData.reshape(trainData.shape()[0], -1);
        testData = testData.reshape(testData.shape()[0], -1);

        System.out.println("Train data shape: " + trainData.shapeInfoToString());
        System.out.println("Test data shape: " + testData.shapeInfoToString());
    }

    public DataLoader(String trainX, String trainY, String testX, String testY) throws IOException {
        File trainXFile = new File(trainX);
        File trainYFile = new File(trainY);
        File testXFile = new File(testX);
        File testYFile = new File(testY);

        if (trainXFile.exists() && trainYFile.exists() && testXFile.exists() && testYFile.exists()) {
            INDArray xTrain = Nd4j.createFromNpyFile(trainXFile);
            INDArray yTrain = Nd4j.createFromNpyFile(trainYFile);
            INDArray xTest = Nd4j.createFromNpyFile(testXFile);
            INDArray yTest = Nd4j.createFromNpyFile(testYFile);

            // Flatten the data
            xTrain = xTrain.reshape(xTrain.shape()[0], -1);
            xTest = xTest.reshape(xTest.shape()[0], -1);
            yTrain = yTrain.reshape(yTrain.shape()[0], -1);
            yTest = yTest.reshape(yTest.shape()[0], -1);

            System.out.println("xTrain shape: " + xTrain.shapeInfoToString());
            System.out.println("yTrain shape: " + yTrain.shapeInfoToString());
            System.out.println("xTest shape: " + xTest.shapeInfoToString());
            System.out.println("yTest shape: " + yTest.shapeInfoToString());

            // Concatenate x and y
            trainData = Nd4j.concat(1, yTrain, xTrain);
            testData = Nd4j.concat(1, yTest, xTest);
        } else {
            throw new IOException("Data files not found");
        }

        System.out.println("Train data shape: " + trainData.shapeInfoToString());
        System.out.println("Test data shape: " + testData.shapeInfoToString());
    }

    public static INDArray loadCsv(String csvFile) {
        List<double[]> data = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(csvFile))) {
            String line;

            // Skip the first line (header row)
            reader.readLine();
            while ((line = reader.readLine()) != null) {
                String[] tokens = line.split(",");
                double[] row = new double[tokens.length];
                for (int i = 0; i < tokens.length; i++) {
                    row[i] = Double.parseDouble(tokens[i]);
                }
                data.add(row);
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            System.out.println("Loaded " + data.size() + " rows from " + csvFile);
        }

        return Nd4j.create(data.toArray(new double[0][]));
    }

    public INDArray getAllTrainImages() {
        return trainData.get(NDArrayIndex.all(), NDArrayIndex.interval(1, trainData.columns()));
    }

    public INDArray getAllTestImages() {
        return testData.get(NDArrayIndex.all(), NDArrayIndex.interval(1, testData.columns()));
    }

    public INDArray getAllTrainLabels() {
        return trainData.get(NDArrayIndex.all(), NDArrayIndex.point(0)).reshape(trainData.rows(), 1);
    }

    public INDArray getAllTestLabels() {
        return testData.get(NDArrayIndex.all(), NDArrayIndex.point(0)).reshape(testData.rows(), 1);
    }

    public INDArray getTrainImage(int index) {
        return trainData.getRow(index).get(NDArrayIndex.interval(1, trainData.columns()));
    }

    public INDArray getTestImage(int index) {
        return testData.getRow(index).get(NDArrayIndex.interval(1, testData.columns()));
    }

    public int getTrainLabel(int index) {
        return (int) trainData.getRow(index).getDouble(0);
    }

    public int getTestLabel(int index) {
        return (int) testData.getRow(index).getDouble(0);
    }

    public INDArray getTrainData() {
        return trainData;
    }

    public INDArray getTestData() {
        return testData;
    }
}