package br.cafedl.game.controller;

import br.cafedl.game.model.MyCNNModel;
import br.cafedl.game.model.PredictionModel;
import br.cafedl.game.model.database.PersistenceManager;
import br.cafedl.game.viewmodel.ViewModelManager;
import javafx.event.ActionEvent;
import javafx.fxml.FXMLLoader;
import javafx.scene.Node;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;

import jakarta.persistence.EntityManager;
import java.io.IOException;

public class MenuController {
    private static final EntityManager gameEntityManager;
    private static final PredictionModel model;
    static {
        gameEntityManager = PersistenceManager.createEntityManager("quickdrawPU");

        model = new MyCNNModel();
        model.loadModel();
    }

    public void handleStartButtonAction(ActionEvent actionEvent) throws IOException {

        ViewModelManager.getInstance().startNewSession(model, gameEntityManager);

        Stage stage = (Stage) ((Node) actionEvent.getSource()).getScene().getWindow();
        FXMLLoader loader = new FXMLLoader(MenuController.class.getResource("fxml/transition-view.fxml"));
        Parent root = loader.load();
        TransitionController controller = loader.getController();
        controller.setViewModel(ViewModelManager.getInstance().getTransitionViewModel(controller));
        Scene scene = new Scene(root);
        stage.setScene(scene);
        stage.show();
    }

    public void handleDrawsButtonAction(ActionEvent actionEvent) throws IOException {
        Stage stage = (Stage) ((Node) actionEvent.getSource()).getScene().getWindow();
        FXMLLoader loader = new FXMLLoader(MenuController.class.getResource("fxml/drawings-view.fxml"));
        Parent root = loader.load();
        DrawingsController controller = loader.getController();
        Scene scene = new Scene(root);
        stage.setScene(scene);
        stage.show();
    }
}
