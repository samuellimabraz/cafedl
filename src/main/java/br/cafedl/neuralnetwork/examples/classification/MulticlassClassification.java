package br.cafedl.neuralnetwork.examples.classification;

import br.cafedl.neuralnetwork.core.activation.Activation;
import br.cafedl.neuralnetwork.core.layers.Dense;
import br.cafedl.neuralnetwork.core.losses.SoftmaxCrossEntropy;
import br.cafedl.neuralnetwork.core.metrics.Accuracy;
import br.cafedl.neuralnetwork.core.metrics.F1Score;
import br.cafedl.neuralnetwork.core.metrics.Precision;
import br.cafedl.neuralnetwork.core.metrics.Recall;
import br.cafedl.neuralnetwork.core.models.ModelBuilder;
import br.cafedl.neuralnetwork.core.models.NeuralNetwork;
import br.cafedl.neuralnetwork.core.optimizers.Optimizer;
import br.cafedl.neuralnetwork.core.optimizers.RMSProp;
import br.cafedl.neuralnetwork.core.train.Trainer;
import br.cafedl.neuralnetwork.core.train.TrainerBuilder;
import br.cafedl.neuralnetwork.data.Util;
import br.cafedl.neuralnetwork.data.processing.DataProcessor;
import br.cafedl.neuralnetwork.data.processing.StandardScaler;
import br.cafedl.neuralnetwork.data.PlotDataPredict;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

public class MulticlassClassification {
    public static void main(String[] args) {

        // Criação dos dados de treinamento
        int numSamples = 200;
        INDArray x = Nd4j.linspace(numSamples / 2 * -1, numSamples / 2, numSamples).reshape(numSamples, 1).castTo(DataType.DOUBLE);
        // Classificação multiclasse baseada em qual quartil x pertence
        INDArray y = Nd4j.zeros(numSamples, 4);
        for (int i = 0; i < numSamples; i++) {
            int quartile = (int) Math.abs(x.getDouble(i)) / 25;
            if (quartile == 4) quartile = 3;
            y.putScalar(new int[]{i, quartile}, 1.0);
        }

        // Normalização dos dados
        DataProcessor scaler = new StandardScaler();  // Normalização min-max
        INDArray x_scaled = scaler.fitTransform(x);

        // Shapes
        System.out.println("x_scaled shape: " + Arrays.toString(x_scaled.shape()));
        System.out.println("y shape: " + Arrays.toString(y.shape()));

        // Criação do modelo
        NeuralNetwork model = new ModelBuilder()
                .add(new Dense(16, Activation.create("relu"), "he"))
                .add(new Dense(16, Activation.create("relu"), "he"))
                .add(new Dense(4, Activation.create("linear"), "xavier"))
                .build();

        x_scaled.castTo(DataType.DOUBLE);
        y.castTo(DataType.DOUBLE);

        // Treinamento do modelo
        Optimizer optimizer = new RMSProp(0.001);
        Trainer trainer = new TrainerBuilder(model, x_scaled, y,  new SoftmaxCrossEntropy())
                .setOptimizer(optimizer)
                .setEpochs(200)
                .setBatchSize(32)
                .setEarlyStopping(true)
                .setTrainRatio(1.0)
                .setPatience(5)
                .setEvalEvery(10)
                .setMetric(new Accuracy())
                .build();

        trainer.fit();

        // Predição
        INDArray x_test = trainer.getTestInputs();
        INDArray y_test = trainer.getTestTargets();


        INDArray predict = model.predict(x_test);
        x_test = scaler.inverseTransform(x_test);

        INDArray predictClass = predict.argMax(1);
        INDArray y_testClass = y_test.argMax(1);


        // Comparação entre os valores reais e os valores previstos
        INDArray comp = predictClass.eq(y_testClass).castTo(Nd4j.defaultFloatingPointType());
        System.out.println("\nPredict vs y_test: ");
        System.out.println("True: " + comp.sumNumber().intValue());
        System.out.println("False: " + (y_testClass.length() - comp.sumNumber().intValue()));

        INDArray predictOneHot = Util.oneHotEncode(predictClass.reshape(predictClass.length(), 1), 4);

        System.out.println("Predict: " + predict);
        System.out.println("Predict argmax: " + predictClass);

        INDArray percentages = predict.mul(100);
        System.out.println("Example 0: " + percentages.getRow(0));
        System.out.println("Example 1: " + percentages.getRow(0));

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

        // Teste de predição
        INDArray test = Nd4j.create(new double[]{10, 30, 50, 70, 90}, new int[]{5, 1});
        INDArray test_predict = model.predict(scaler.transform(test));
        System.out.println("Test predict: " + test_predict);
        System.out.println("Test predict argmax: " + test_predict.argMax(1));

        // Plotagem dos dados
        PlotDataPredict plotDataPredict = new PlotDataPredict();
        plotDataPredict.plot2d(x_test, y_testClass, predictClass, "Classificação Multiclasse");
    }
}
