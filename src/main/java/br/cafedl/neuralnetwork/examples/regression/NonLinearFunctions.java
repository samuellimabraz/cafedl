package br.cafedl.neuralnetwork.examples.regression;

import br.cafedl.neuralnetwork.core.activation.Activation;
import br.cafedl.neuralnetwork.core.layers.Dense;
import br.cafedl.neuralnetwork.core.losses.MeanSquaredError;
import br.cafedl.neuralnetwork.core.metrics.MAE;
import br.cafedl.neuralnetwork.core.metrics.MSE;
import br.cafedl.neuralnetwork.core.metrics.R2;
import br.cafedl.neuralnetwork.core.metrics.RMSE;
import br.cafedl.neuralnetwork.core.models.ModelBuilder;
import br.cafedl.neuralnetwork.core.models.NeuralNetwork;
import br.cafedl.neuralnetwork.core.optimizers.*;
import br.cafedl.neuralnetwork.core.train.Trainer;
import br.cafedl.neuralnetwork.core.train.TrainerBuilder;
import br.cafedl.neuralnetwork.data.processing.DataProcessor;
import br.cafedl.neuralnetwork.data.processing.StandardScaler;
import br.cafedl.neuralnetwork.data.PlotDataPredict;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Arrays;
import java.util.Scanner;

public class NonLinearFunctions {

    public void sineFunction() {
        // Criação dos dados de treinamento
        int numSamples = 250;
        INDArray x = Nd4j.linspace((long) (-2*Math.PI), (long) (2*Math.PI), numSamples, DataType.DOUBLE).reshape(numSamples, 1);
        INDArray y = Transforms.sin(x);

        System.out.println("x type: " + x.dataType());
        System.out.println("y type: " + y.dataType());

        System.out.println("x shape: " + Arrays.toString(x.shape()));
        System.out.println("y shape: " + Arrays.toString(y.shape()));

        // Normalização dos dados
//        DataProcessor scaler = new StandardScaler();
//        INDArray x_scaled = scaler.fitTransform(x);

        // Criação do modelo
        NeuralNetwork model = new ModelBuilder()
                .add(new Dense(16, Activation.create("relu"), "he"))
                .add(new Dense(16, Activation.create("relu"), "he"))
                .add(new Dense(1, Activation.create("tanh"), "xavier"))
                .build();

        // Treinamento do modelo
        Optimizer optimizer = new Adam(0.001);
        Trainer trainer = new TrainerBuilder(model, x, y, new MeanSquaredError())
                .setOptimizer(optimizer)
                .setEpochs(200)
                .setBatchSize(32)
                .setTrainRatio(1.0)
                .setEarlyStopping(true)
                .setPatience(10)
                .setEvalEvery(10)
                .build();

        INDArray x_test = trainer.getTestInputs();
        INDArray y_test = trainer.getTestTargets();

        trainer.fit();

        // Predição

        INDArray predict = model.predict(x_test);
//        x_test = scaler.inverseTransform(x_test);

        printMetrics(y_test, predict);

        Dense layer = (Dense) model.getLayers().get(0);
        System.out.println("Dense weights layer 0: " + layer.getWeights());
        System.out.println("Dense bias layer 0: " + layer.getBias());

        // Criação do gráfico
        PlotDataPredict plotDataPredict = new PlotDataPredict();
        plotDataPredict.plot2d(x_test, y_test, predict, "Função Seno");

        try {
            model.saveModel("src/main/resources/br/deeplearning4java/neuralnetwork/examples/data/regression/sine_function.zip");
            System.out.println("Modelo salvo com sucesso!");
        } catch (Exception e) {
            System.out.println("Erro ao salvar o modelo!");
        }
    }

    public void squareFunction() {
        // Criação dos dados de treinamento para a função quadrática
        int numSamples = 100;
        INDArray x = Nd4j.linspace(-10, 10, numSamples, DataType.DOUBLE).reshape(numSamples, 1);
        INDArray y = x.mul(x);

        // Normalização dos dados
        DataProcessor scaler = new StandardScaler();
        INDArray x_scaled = scaler.fitTransform(x);

        DataProcessor scaler2 = new StandardScaler();
        INDArray y_scaled = scaler2.fitTransform(y);

        // Criação do modelo
        NeuralNetwork model = new ModelBuilder()
                .add(new Dense(32, Activation.create("relu"), "he"))
                .add(new Dense(32, Activation.create("relu"), "he"))
                .add(new Dense(1, Activation.create("linear"), "xavier"))
                .build();

        // Treinamento do modelo
        Optimizer optimizer = new Adam(0.001);
        Trainer trainer = new TrainerBuilder(model, x_scaled, y_scaled, new MeanSquaredError())
                .setOptimizer(optimizer)
                .setEpochs(200)
                .setBatchSize(25)
                .setTrainRatio(1.0)
                .setEarlyStopping(true)
                .setPatience(30)
                .build();
        trainer.fit();


        // Predição
        INDArray x_test = trainer.getTestInputs();
        INDArray y_test = trainer.getTestTargets();
        INDArray predict = model.predict(x_test);
        predict = scaler2.inverseTransform(predict);
        x_test = scaler.inverseTransform(x_test);
        y_test = scaler2.inverseTransform(y_test);

        printMetrics(y_test, predict);

        // Criação do gráfico
        PlotDataPredict plotDataPredict = new PlotDataPredict();
        plotDataPredict.plot2d(x_test, y_test, predict, "Função Quadrática");

    }

    public void cubicFunction() {
        // Criação dos dados de treinamento para a função cúbica
        int numSamples = 120;
        INDArray x = Nd4j.linspace(-10, 10, numSamples, DataType.DOUBLE).reshape(numSamples, 1);
        INDArray y = x.mul(x).mul(x);

        System.out.println("x: " + Arrays.toString(x.data().asFloat()));
        System.out.println("y: " + Arrays.toString(y.data().asFloat()));

        // Normalização dos dados
        DataProcessor scaler = new StandardScaler();
        INDArray x_scaled = scaler.fitTransform(x);

        DataProcessor scaler2 = new StandardScaler();
        INDArray y_scaled = scaler2.fitTransform(y);

        // Criação do modelo
        NeuralNetwork model = new ModelBuilder()
                .add(new Dense(10, Activation.create("relu"), "xavier"))
                .add(new Dense(16, Activation.create("tanh"), "xavier"))
                .add(new Dense(1, Activation.create("linear"), "xavier"))
                .build();

        // Treinamento do modelo
        Optimizer optimizer = new SGDMomentum(0.00026, 0.9);
        Trainer trainer = new TrainerBuilder(model, x_scaled, y_scaled, new MeanSquaredError())
                .setOptimizer(optimizer)
                .setEpochs(2000)
                .setBatchSize(20)
                .setTrainRatio(1.0)
                .setEarlyStopping(true)
                .setPatience(30)
                .setEvalEvery(25)
                .build();

        long startTime = System.nanoTime();
        trainer.fit();
        long endTime = System.nanoTime();

        double durationInSeconds = (endTime - startTime) / 1_000_000_000.0;
        System.out.println("Duration: " + durationInSeconds + " seconds");

        // Predição
        INDArray x_test = trainer.getTestInputs();
        INDArray y_test = trainer.getTestTargets();
        INDArray predict = model.predict(x_test);
        predict = scaler2.inverseTransform(predict);
        x_test = scaler.inverseTransform(x_test);
        y_test = scaler2.inverseTransform(y_test);

        printMetrics(y_test, predict);

        // Criação do gráfico
        PlotDataPredict plotDataPredict = new PlotDataPredict();
        plotDataPredict.plot2d(x_test, y_test, predict, "Função Cúbica");
    }

    public void saddleFunction() {
        // Crei um conjunto de dados aleatórios
        int numSamples = 1000;
        INDArray x = Nd4j.rand(DataType.DOUBLE, numSamples, 2).mul(10).sub(5);
        INDArray y = saddle(x).reshape(numSamples, 1);

//        System.out.println("x: " + Arrays.toString(x.data().asFloat()));
//        System.out.println("y: " + Arrays.toString(y.data().asFloat()));
        System.out.println("x shape: " + Arrays.toString(x.shape()));
        System.out.println("y shape: " + Arrays.toString(y.shape()));
        System.out.println("X range: " + x.minNumber() + " - " + x.maxNumber());

        // Normalização dos dados
        DataProcessor scaler = new StandardScaler();
        INDArray x_scaled = scaler.fitTransform(x);

        DataProcessor scaler2 = new StandardScaler();
        INDArray y_scaled = scaler2.fitTransform(y);

        // Criação do modelo
        NeuralNetwork model = new ModelBuilder()
                .add(new Dense(89, Activation.create("relu"), "he"))
                .add(new Dense(16, Activation.create("tanh"), "xavier"))
                .add(new Dense(1, Activation.create("linear")))
                .build();

        int epochs = 20;
        int batchSize = 64;

        // Treinamento do modelo
        LearningRateDecayStrategy lr = new ExponentialDecayStrategy(0.001, 0.0001, epochs);
        Optimizer optimizer = new RMSProp(lr);
        Trainer trainer = new TrainerBuilder(model, x_scaled, y_scaled, new MeanSquaredError())
                .setOptimizer(optimizer)
                .setEpochs(100)
                .setBatchSize(batchSize)
                .setTrainRatio(1.0)
                .setEarlyStopping(true)
                .setPatience(5)
                .setEvalEvery(10)
                .build();

        trainer.printDataInfo();
        trainer.fit();


        // Predição
        INDArray x_test = trainer.getTestInputs();
        INDArray y_test = trainer.getTestTargets();
        INDArray predict = model.predict(x_test);
        predict = scaler2.inverseTransform(predict).reshape(y_test.shape());
        x_test = scaler.inverseTransform(x_test);
        y_test = scaler2.inverseTransform(y_test).reshape(y_test.shape());

        printMetrics(y_test, predict);
        // Criação do gráfico
        PlotDataPredict plotDataPredict = new PlotDataPredict();
        plotDataPredict.plot3dGridandScatter(x_test, y_test, predict, "Função Saddle");

        try {
            model.setName("saddle_function");
            model.saveModel("src/main/resources/br/deeplearning4java/neuralnetwork/examples/data/regression/saddle_function_model.zip");
            System.out.println("Modelo salvo com sucesso!");
        } catch (Exception e) {
            System.out.println("Erro ao salvar o modelo!");
        }
    }

    public INDArray saddle(INDArray x) {
        return x.getColumn(0).mul(x.getColumn(0)).sub(x.getColumn(1).mul(x.getColumn(1)));
    }

    public void scweffelFunction() {
        // Crei um conjunto de dados aleatórios
        int numSamples = 500;
        INDArray x = Nd4j.rand(DataType.DOUBLE, numSamples, 2).mul(800).sub(400);
        INDArray y = scweffel(x).reshape(numSamples, 1);

        System.out.println("X range: " + x.minNumber() + " - " + x.maxNumber());

        // Normalização dos dados
        DataProcessor scaler = new StandardScaler();
        INDArray x_scaled = scaler.fitTransform(x);

        DataProcessor scaler2 = new StandardScaler();
        INDArray y_scaled = scaler2.fitTransform(y);

        // Criação do modelo
        NeuralNetwork model = new ModelBuilder()
                .add(new Dense(32, Activation.create("relu"), "he"))
                .add(new Dense(32, Activation.create("relu"), "he"))
                .add(new Dense(16, Activation.create("relu"), "he"))
                .add(new Dense(1, Activation.create("linear"), "xavier"))
                .build();

        // Treinamento do modelo
        Optimizer optimizer = new RMSProp(0.0015);
        Trainer trainer = new TrainerBuilder(model, x_scaled, y_scaled, new MeanSquaredError())
                .setOptimizer(optimizer)
                .setEpochs(500)
                .setBatchSize(32)
                .setTrainRatio(1.0)
                .setEarlyStopping(true)
                .setPatience(5)
                .setEvalEvery(10)
                .build();

        trainer.printDataInfo();

        trainer.fit();


        // Predição
        INDArray x_test = trainer.getTestInputs();
        INDArray y_test = trainer.getTestTargets();
        INDArray predict = model.predict(x_test);
        x_test = scaler.inverseTransform(x_test);
        y_test = scaler2.inverseTransform(y_test);
        predict = scaler2.inverseTransform(predict);

        printMetrics(y_test, predict);

        // Criação do gráfico
        PlotDataPredict plotDataPredict = new PlotDataPredict();
        plotDataPredict.plot3dGridandScatter(x_test, y_test, predict, "Função Scweffel");
    }

    public INDArray scweffel(INDArray x) {
        return x.getColumn(0).mul(Transforms.sin(Transforms.sqrt(Transforms.abs(x.getColumn(0))))).sub(x.getColumn(1).mul(Transforms.sin(Transforms.sqrt(Transforms.abs(x.getColumn(1)))))).add(-418.9829 * 2);
    }

    public void rosenbrockFunction() {
        // Crei um conjunto de dados aleatórios
        int numSamples = 700;
        // x1 = [-2, 2]
        // x2 = [-1, 3]
        INDArray x1 = Nd4j.rand(DataType.DOUBLE, numSamples, 1).mul(4).sub(2);
        INDArray x2 = Nd4j.rand(DataType.DOUBLE,numSamples, 1).mul(4).sub(1);
        INDArray x = Nd4j.concat(1, x1, x2);
        INDArray y = rosenbrock(x).reshape(numSamples, 1);

        System.out.println("x1 range: " + x1.minNumber() + " - " + x1.maxNumber());
        System.out.println("x2 range: " + x2.minNumber() + " - " + x2.maxNumber());
        System.out.println("x shape: " + Arrays.toString(x.shape()));
        System.out.println("y shape: " + Arrays.toString(y.shape()));

        // Normalização dos dados
        DataProcessor scaler = new StandardScaler();
        INDArray x_scaled = scaler.fitTransform(x);

        DataProcessor scaler2 = new StandardScaler();
        INDArray y_scaled = scaler2.fitTransform(y);


        // Criação do modelo
        NeuralNetwork model = new ModelBuilder()
                .add(new Dense(16, Activation.create("relu"), "he"))
                .add(new Dense(32, Activation.create("relu"), "he"))
                .add(new Dense(1, Activation.create("linear"), "xavier"))
                .build();

        // Treinamento do modelo
        Optimizer optimizer = new SGDNesterov(0.00013, 0.9);//new Adam(0.00012, 0.9, 0.999, 1e-8);
        Trainer trainer = new TrainerBuilder(model, x_scaled, y_scaled, new MeanSquaredError())
                .setOptimizer(optimizer)
                .setEpochs(300)
                .setBatchSize(45)
                .setTrainRatio(1.0)
                .setEarlyStopping(true)
                .setPatience(200)
                .setEvalEvery(50)
                .build();

        trainer.printDataInfo();
        trainer.fit();


        // Predição
        INDArray x_test = trainer.getTestInputs();
        INDArray y_test = trainer.getTestTargets();
        INDArray predict = model.predict(x_test);
        predict = scaler2.inverseTransform(predict);
        x_test = scaler.inverseTransform(x_test);
        y_test = scaler2.inverseTransform(y_test);

        // R2 Score
        R2 r2 = new R2();
        System.out.println("R2 Score: " + r2.evaluate(y_test, predict));

        // Criação do gráfico
        PlotDataPredict plotDataPredict = new PlotDataPredict();
        plotDataPredict.plot3dGridandScatter(x_test, y_test, predict, "Função Rosenbrock");
    }

    public INDArray rosenbrock(INDArray x) {
        // a = 1
        // b = 100
        // return (a - x) ** 2 + b * (y - x ** 2) ** 2
        return Transforms.pow(x.getColumn(0).rsub(1.0), 2).add(Transforms.pow((x.getColumn(1).subi(Transforms.pow(x.getColumn(0), 2))), 2).mul(100));
    }

    public void printMetrics(INDArray y, INDArray predicts) {
        R2 r2 = new R2();
        MAE mea = new MAE();
        RMSE rmse = new RMSE();
        MSE mse = new MSE();
        System.out.println("MAE: " + mea.evaluate(y, predicts));
        System.out.println("RMSE: " + rmse.evaluate(y, predicts));
        System.out.println("MSE: " + mse.evaluate(y, predicts));
        System.out.println("R2 Score: " + r2.evaluate(y, predicts));
    }

    public static void main(String[] args) {
        NonLinearFunctions nonLinearFunctions = new NonLinearFunctions();
        System.out.println("Select the function to be fitted:");
        System.out.println("1 - Sine Function");
        System.out.println("2 - Square Function");
        System.out.println("3 - Cubic Function");
        System.out.println("4 - Saddel Function");
        System.out.println("5 - Scweffel Function");
        System.out.println("6 - Rosenbrock Function");
        System.out.println("7 - Exit");

            Scanner scanner = new Scanner(System.in);
            int option = 0;
            while (option != 5) {
                option = Integer.parseInt(scanner.nextLine());
                switch (option) {
                    case 1:
                        nonLinearFunctions.sineFunction();
                        break;
                    case 2:
                        nonLinearFunctions.squareFunction();
                        break;
                    case 3:
                        nonLinearFunctions.cubicFunction();
                        break;
                    case 4:
                        nonLinearFunctions.saddleFunction();
                        break;
                    case 5:
                        nonLinearFunctions.scweffelFunction();
                        break;
                    case 6:
                        nonLinearFunctions.rosenbrockFunction();
                        break;
                    case 7:
                        System.out.println("Exiting...");
                        break;
                    default:
                        System.out.println("Invalid option!");
                        break;
                }
            }
            scanner.close();
    }

}
