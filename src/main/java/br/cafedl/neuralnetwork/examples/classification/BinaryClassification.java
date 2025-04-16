package br.cafedl.neuralnetwork.examples.classification;

import br.cafedl.neuralnetwork.core.activation.Activation;
import br.cafedl.neuralnetwork.core.layers.Dense;
import br.cafedl.neuralnetwork.core.losses.BinaryCrossEntropy;
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
import br.cafedl.neuralnetwork.data.PlotDataPredict;
import br.cafedl.neuralnetwork.data.processing.DataProcessor;
import br.cafedl.neuralnetwork.data.processing.StandardScaler;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

public class BinaryClassification {
    public static void main(String[] args) {
        // Criação dos dados de treinamento
        int numSamples = 100;
        INDArray x = Nd4j.linspace(DataType.DOUBLE, 1.0, 1.0, numSamples).reshape(numSamples, 1);
        System.out.println("x range: " + x.minNumber().doubleValue() + " - " + x.maxNumber().doubleValue());
        // Classificação binária baseada em se x é maior que 50
        INDArray y = x.gt(50.0).castTo(DataType.DOUBLE);

        System.out.println("y: " + y);

        // Normalização dos dados
        DataProcessor scaler = new StandardScaler();
        INDArray x_scaled = scaler.fitTransform(x);


        System.out.println("x shape: " + Arrays.toString(x.shape()));
        System.out.println("y shape: " + Arrays.toString(y.shape()));

        // Criação do modelo
        NeuralNetwork model = new ModelBuilder()
                .add(new Dense(1, Activation.create("sigmoid")))  // Função de ativação sigmoid para classificação binária
                .build();

        // Treinamento do modelo
        Optimizer optimizer = new RMSProp(0.007);
        Trainer trainer = new TrainerBuilder(model, x_scaled, y, new BinaryCrossEntropy())
                .setOptimizer(optimizer)
                .setEpochs(500)
                .setBatchSize(10)
                .setTrainRatio(1.0)
                .setEarlyStopping(true)
                .setPatience(30)
                .build();
        trainer.fit();
        // Predição
        INDArray x_test = trainer.getTestInputs();
        INDArray y_test = trainer.getTestTargets();
        INDArray predict = model.predict(x_test);
        predict = predict.gt(0.5).castTo(y.dataType());  // Classificação binária baseada em se a previsão é maior que 0.5
        x_test = scaler.inverseTransform(x_test);


        // Comparação entre os valores reais e os valores previstos
        System.out.println("Predict vs y_test: " + predict.eq(y_test));

        // Metricas
        Accuracy accuracy = new Accuracy();
        double acc = accuracy.evaluate(y_test, predict);
        System.out.println("Accuracy: " + acc);
        Precision precision = new Precision();
        double prec = precision.evaluate(y_test, predict);
        System.out.println("Precision: " + prec);
        Recall recall = new Recall();
        double rec = recall.evaluate(y_test, predict);
        System.out.println("Recall: " + rec);
        F1Score f1Score = new F1Score();
        double f1 = f1Score.evaluate(y_test, predict);
        System.out.println("F1 Score: " + f1);

        // Teste de predição x > 50 == 1 (true) ou x <= 50 == 0 (false)
        INDArray test = Nd4j.create(new double[]{51, 32, 53, 4, 55, 49, 57, 40, 59, 60}, new int[]{10, 1});
        INDArray test_predict = model.predict(scaler.transform(test));
        test_predict = test_predict.gt(0.5).castTo(y.dataType());
        System.out.println("Test predict: " + test_predict);

        // Teste de predição x > 50 == 1 (true) ou x <= 50 == 0 (false)
        INDArray test2 = Nd4j.create(new double[]{1, 2, 3, 60, 5, 6, 7, 51, 9, 10}, new int[]{10, 1});
        INDArray test_predict2 = model.predict(scaler.transform(test2));
        test_predict2 = test_predict2.gt(0.5).castTo(y.dataType());
        System.out.println("Test predict: " + test_predict2);

        // Criação do gráfico
        PlotDataPredict plotDataPredict = new PlotDataPredict();
        plotDataPredict.plot2d(x_test, y_test, predict, "Regressão Logistica");
    }
}

