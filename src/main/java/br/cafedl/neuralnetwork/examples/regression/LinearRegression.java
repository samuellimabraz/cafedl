package br.cafedl.neuralnetwork.examples.regression;

import br.cafedl.neuralnetwork.core.activation.Activation;
import br.cafedl.neuralnetwork.core.layers.Dense;
import br.cafedl.neuralnetwork.core.losses.MeanSquaredError;
import br.cafedl.neuralnetwork.core.metrics.Accuracy;
import br.cafedl.neuralnetwork.core.metrics.MAE;
import br.cafedl.neuralnetwork.core.metrics.RMSE;
import br.cafedl.neuralnetwork.core.models.ModelBuilder;
import br.cafedl.neuralnetwork.core.models.NeuralNetwork;
import br.cafedl.neuralnetwork.core.optimizers.Optimizer;
import br.cafedl.neuralnetwork.core.optimizers.SGDMomentum;
import br.cafedl.neuralnetwork.core.train.Trainer;
import br.cafedl.neuralnetwork.core.train.TrainerBuilder;
import br.cafedl.neuralnetwork.data.processing.DataProcessor;
import br.cafedl.neuralnetwork.data.processing.StandardScaler;
import br.cafedl.neuralnetwork.data.PlotDataPredict;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class LinearRegression {
    public static void main(String[] args) {
        // Criação dos dados de treinamento
        int numSamples = 300;
        INDArray x = Nd4j.linspace(1, numSamples, numSamples).reshape(numSamples, 1).castTo(DataType.DOUBLE);
        INDArray y = x.mul(2).add(40).add(Nd4j.randn(numSamples, 1).mul(7)).castTo(DataType.DOUBLE);

        // Normalização dos dados
        DataProcessor scaler = new StandardScaler();
        x = scaler.fitTransform(x);

        // Criação do modelo
        NeuralNetwork model = new ModelBuilder()
                .add(new Dense(1, Activation.create("linear"), "he"))
                .build();

        // Treinamento do modelo
        Optimizer optimizer = new SGDMomentum(0.002, 0.9);
        Trainer trainer = new TrainerBuilder(model, x, y, new MeanSquaredError())
                .setOptimizer(optimizer)
                .setEpochs(100)
                .setBatchSize(32)
                .setTrainRatio(0.8)
                .setEarlyStopping(true)
                .setPatience(5)
                .setEvalEvery(2)
                .build();
        trainer.fit();

        // Predição
        INDArray x_test = trainer.getTestInputs();
        INDArray y_test = trainer.getTestTargets();
        INDArray predict = model.predict(x_test);

        // Métricas
        MAE mae = new MAE();
        RMSE rmse = new RMSE();
        Accuracy accuracy = new Accuracy();
        System.out.println("MAE: " + mae.evaluate(y_test, predict));
        System.out.println("RMSE: " + rmse.evaluate(y_test, predict));
        System.out.println("Accuracy: " + accuracy.evaluate(y_test, predict));


        // Criação do gráfico
        PlotDataPredict plotDataPredict = new PlotDataPredict();
        plotDataPredict.plot2d(x_test, y_test, predict, "Regressão Linear");

        // Salva o modelo
        try {
            model.setName("LinearRegression");
            model.saveModel("src/main/resources/br/deeplearning4java/neuralnetwork/examples/data/regression/linear_regression.zip");
            System.out.println("Modelo salvo com sucesso!");
        } catch (Exception e) {
            e.printStackTrace();
        }

        Dense layer = (Dense) model.getTrainableLayers().get(0);
        System.out.println("Params: " + layer.getParams());
        System.out.println("Weights: " + layer.getWeights());
        System.out.println("Bias: " + layer.getBias());
    }
}

