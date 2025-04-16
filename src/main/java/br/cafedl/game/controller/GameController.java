package br.cafedl.game.controller;

import br.cafedl.game.model.PredictionResult;
import br.cafedl.game.viewmodel.GameViewModel;
import br.cafedl.game.viewmodel.ViewModelManager;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.fxml.Initializable;
import javafx.scene.Node;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.SnapshotParameters;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.ListView;
import javafx.scene.image.WritableImage;
import javafx.scene.input.MouseEvent;
import javafx.scene.paint.Color;
import javafx.stage.Stage;

import java.io.IOException;
import java.net.URL;
import java.util.List;
import java.util.ResourceBundle;

public class GameController implements Initializable {

    private GameViewModel viewModel;
    public Label categoryLabel;
    public Label timeLabel;
    public Button eraseButton;
    public Button nextButton;
    public Button closeButton;
    public ListView listPredictions;
    @FXML
    private Canvas canvas;
    private GraphicsContext gc;


    @Override
    public void initialize(URL url, ResourceBundle resourceBundle) {
        System.out.println("GameController initialized");

        gc = canvas.getGraphicsContext2D();
        // Set canvas to gray scale
        gc.setFill(Color.WHITE);
        gc.fillRect(0, 0, canvas.getWidth(), canvas.getHeight());
        gc.setStroke(Color.BLACK);
        gc.setLineWidth(3.5);

        canvas.addEventHandler(MouseEvent.MOUSE_PRESSED, event -> {
            gc.beginPath();
            gc.moveTo(event.getX(), event.getY());
            gc.stroke();
        });

        canvas.addEventHandler(MouseEvent.MOUSE_DRAGGED, event -> {
            gc.lineTo(event.getX(), event.getY());
            gc.stroke();
        });

        canvas.addEventHandler(MouseEvent.MOUSE_RELEASED, event -> {
            if (!canvas.isDisable()) {
                try {
                    SnapshotParameters params = new SnapshotParameters();
                    params.setFill(Color.WHITE);
                    handleDrawing(canvas.snapshot(params, null));
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
                gc.closePath();
            }
        });
    }

    public void disableCanvas(boolean enable) {
        canvas.setDisable(enable);
    }

    public void handleDrawing(WritableImage drawing) throws IOException {
        viewModel.sendDrawing(drawing);
    }

    public void handleEraseButtonAction(ActionEvent actionEvent) {
        gc.setFill(Color.WHITE);
        gc.fillRect(0, 0, canvas.getWidth(), canvas.getHeight());
        gc.setStroke(Color.BLACK);
        gc.setLineWidth(3.5);
    }

    public void handleNextButtonAction(ActionEvent actionEvent) throws IOException {
        viewModel.nextRound(false);
    }

    public void nextRound() throws IOException {
        Node source = nextButton;
        if (source.getScene() != null) {
            Stage stage = (Stage) source.getScene().getWindow();
            FXMLLoader loader = new FXMLLoader(getClass().getResource("fxml/transition-view.fxml"));
            Parent root = loader.load();
            TransitionController controller = loader.getController();
            controller.setViewModel(ViewModelManager.getInstance().getTransitionViewModel(controller));
            Scene scene = new Scene(root);
            stage.setScene(scene);
            stage.show();
        }
    }

    public void handleCloseButtonAction(ActionEvent actionEvent) throws IOException {
        viewModel.endSessionGame();
        endGame();
    }

    public void updateTime(int time) {
        timeLabel.setText(String.format("%02d:%02d", time / 60, time % 60));
    }

    public void updateCategory(String category) {
        categoryLabel.setText("Draw: " + category);
    }

    public void updatePredictions(List<PredictionResult> predictions) {
        listPredictions.getItems().clear();
        for (PredictionResult prediction : predictions) {
            listPredictions.getItems().add(prediction.getCategory() + ": " + String.format("%.3f", prediction.getProbability()) + "%");
        }
        listPredictions.refresh();
    }

    public void setViewModel(GameViewModel gameViewModel) {
        this.viewModel = gameViewModel;
        viewModel.updateView();
    }

    public void endGame() throws IOException {
        // Send the image for database
        //viewModel.sendDrawing(canvas.snapshot(null, null));
        // Return to menu equals close button
        Node source = closeButton;
        if (source.getScene() != null) {
            Stage stage = (Stage) source.getScene().getWindow();
            FXMLLoader loader = new FXMLLoader(getClass().getResource("fxml/menu-view.fxml"));
            Parent root = loader.load();
            Scene scene = new Scene(root);
            stage.setScene(scene);
            stage.show();
        }
    }
}