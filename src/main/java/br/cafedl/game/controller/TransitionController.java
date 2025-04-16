package br.cafedl.game.controller;

import br.cafedl.game.viewmodel.TransitionViewModel;
import br.cafedl.game.viewmodel.ViewModelManager;
import javafx.event.ActionEvent;
import javafx.fxml.FXMLLoader;
import javafx.fxml.Initializable;
import javafx.scene.Node;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.stage.Stage;

import java.io.IOException;
import java.net.URL;
import java.util.ResourceBundle;

public class TransitionController implements Initializable {

    private TransitionViewModel viewModel;
    public Label roundLabel;
    public Label instructionLabel;
    public Button confirmButton;
    public Label category;
    public Label drawLabel;

    public void setRound(int round) {
        roundLabel.setText("Draw " + round + " of 6");
    }

    public void setCategory(String category) {
        this.category.setText(category);
    }
    /**
     * @param url
     * @param resourceBundle
     */
    @Override
    public void initialize(URL url, ResourceBundle resourceBundle) {
        System.out.println("TransitionController initialized");
    }

    public void handleStartButtonAction(ActionEvent actionEvent) throws IOException {
        Stage stage = (Stage) ((Node) actionEvent.getSource()).getScene().getWindow();
        FXMLLoader loader = new FXMLLoader(getClass().getResource("fxml/game-view.fxml"));
        Parent root = loader.load();
        GameController controller = loader.getController();
        controller.setViewModel(ViewModelManager.getInstance().getGameViewModel(controller));
        Scene scene = new Scene(root);
        stage.setScene(scene);
        stage.show();
        viewModel.startRound();
    }

    public void setViewModel(TransitionViewModel transitionViewModel) {
        this.viewModel = transitionViewModel;
        viewModel.updateView();
    }
}