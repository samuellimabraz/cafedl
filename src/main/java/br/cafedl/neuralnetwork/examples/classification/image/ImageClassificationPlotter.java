package br.cafedl.neuralnetwork.examples.classification.image;

import br.cafedl.neuralnetwork.core.activation.Activation;
import br.cafedl.neuralnetwork.core.activation.IActivation;
import br.cafedl.neuralnetwork.core.models.NeuralNetwork;
import br.cafedl.neuralnetwork.data.DataLoader;
import br.cafedl.neuralnetwork.data.Util;
import br.cafedl.neuralnetwork.database.NeuralNetworkService;
import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.control.Label;
import javafx.scene.image.ImageView;
import javafx.scene.image.WritableImage;
import javafx.scene.layout.HBox;
import javafx.stage.Stage;
import org.nd4j.linalg.api.ndarray.INDArray;

public class ImageClassificationPlotter extends Application {
    private static final int WIDTH = 28;
    private static final int HEIGHT = 28;
    private static final int SCALE = 10;
    private int currentImageIndex = 0;
    private INDArray imageArray;
    private INDArray imageSample;
    private INDArray predictions;
    private String modelPath;
    private DataLoader dataLoader;
    private NeuralNetwork model;
    private IActivation softmax = Activation.create("softmax");

    @Override
    public void init() throws Exception {
        String example = "quickdraw";

        if (example == "mnist") {
            String dataRoot = "src/main/resources/br/deeplearning4java/neuralnetwork/examples/data/mnist";

            dataLoader = new DataLoader(dataRoot + "/train/mnist_train.bin", dataRoot + "/test/mnist_test.bin");
            model = NeuralNetwork.loadModel(dataRoot + "/model/mnist_model.bin");
        } else if (example == "quickdraw") {
            String dataRoot = "src/main/resources/br/deeplearning4java/neuralnetwork/examples/data/quickdraw";

            dataLoader = new DataLoader(dataRoot + "/npy/train/x_train200.npy", dataRoot + "/npy/train/y_train200.npy", dataRoot + "/npy/test/x_test200.npy", dataRoot + "/npy/test/y_test200.npy");
            NeuralNetworkService service = new NeuralNetworkService();
            model = service.loadModel("quickdraw-cnn");
            model.setInference(true);
        }
    }

    @Override
    public void start(Stage primaryStage) throws Exception {
        primaryStage.setTitle("Image Classification");

        // Obter a primeira imagem
        imageArray = dataLoader.getTestImage(currentImageIndex);
        int label = dataLoader.getTestLabel(currentImageIndex);

        imageSample = imageArray.dup();
        // Normalização (0 - 1)
        imageSample = imageSample.divi(255);
        imageSample = imageSample.reshape(1, 28, 28, 1);

        model.setInference(true);
        predictions = model.predict(imageSample);
        predictions = softmax.forward(predictions.dup());

        System.out.println("Predictions: " + predictions);

        // Converter a imagem para um formato que o JavaFX pode usar
        WritableImage image = Util.arrayToImage(imageArray, WIDTH, HEIGHT);

        // Exibir a imagem
        ImageView imageView = new ImageView(image);
        imageView.setFitWidth(WIDTH * SCALE);  // Ajustar a largura da imagem
        imageView.setFitHeight(HEIGHT * SCALE);  // Ajustar a altura da imagem
        Label labelView = new Label("Label: " + label + " Predictions: " + predictions.argMax(1));
        HBox hbox = new HBox(imageView, labelView);
        Scene scene = new Scene(hbox, WIDTH * SCALE + 200, HEIGHT * SCALE);
        primaryStage.setScene(scene);
        primaryStage.show();

        // Adicionar um manipulador de clique para alternar as imagens
        scene.setOnMouseClicked(event -> {
            currentImageIndex = (currentImageIndex + 1) % dataLoader.getTestData().rows();
            INDArray nextImageArray = dataLoader.getTestImage(currentImageIndex);
            imageSample = nextImageArray.dup();
            // Normalização (0 - 1)
            imageSample = imageSample.divi(255).reshape(1, 28, 28, 1);
            predictions = model.predict(imageSample);
            predictions = softmax.forward(predictions.dup());
            System.out.println("Predictions: " + predictions);

            int nextLabel = dataLoader.getTestLabel(currentImageIndex);
            WritableImage nextImage = Util.arrayToImage(nextImageArray, WIDTH, HEIGHT);
            imageView.setImage(nextImage);
            labelView.setText("Label: " + nextLabel + "\nPredictions: " + predictions.argMax(1) + "\nConfidence: " + predictions.maxNumber().doubleValue() * 100 + "%");
        });
    }


    public void setModelPath(String modelPath) {
        this.modelPath = modelPath;
    }

    public static void main(String[] args) {
        launch(args);
    }
}
