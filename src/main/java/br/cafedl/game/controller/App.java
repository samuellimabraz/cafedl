package br.cafedl.game.controller;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.stage.Stage;
import nu.pattern.OpenCV;

import java.io.IOException;

public class App extends Application {

    @Override
    public void start(Stage stage) throws IOException {
        OpenCV.loadLocally();


        FXMLLoader fxmlLoader = new FXMLLoader(getClass().getResource("fxml/menu-view.fxml"));
        Scene scene = new Scene(fxmlLoader.load(), 770, 620);
        stage.setTitle("Quick Draw!");
        stage.setScene(scene);
        stage.show();

//        stage.setOnCloseRequest(e -> {
//            gameEntityManager.close();
//        });
    }

    public static void main(String[] args) {
        launch(args);
    }
}
