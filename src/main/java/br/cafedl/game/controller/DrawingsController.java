package br.cafedl.game.controller;

import br.cafedl.game.model.Draw;
import br.cafedl.game.model.database.PersistenceManager;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.fxml.Initializable;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.effect.DropShadow;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.stage.Stage;

import javax.persistence.EntityManager;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.net.URL;
import java.util.List;
import java.util.ResourceBundle;

public class DrawingsController implements Initializable  {

    @FXML
    private VBox categoryButtonsVBox;

    @FXML
    private Button backButton;

    @FXML
    private GridPane drawingsGridPane;

    private EntityManager entityManager;

    @Override
    public void initialize(URL location, ResourceBundle resources) {
        entityManager = PersistenceManager.createEntityManager("quickdrawPU");

        loadCategoryButtons();
    }

    private void loadCategoryButtons() {
        List<String> categories = PersistenceManager.getAllCategories(entityManager);

        for (String category : categories) {
            Button button = new Button(category);
            button.setOnAction(event -> loadDrawingsByCategory(category));
            button.setStyle("-fx-text-fill: #f0f0f0; -fx-background-color: #ffd139;");
            button.setEffect(new DropShadow(8.5175, Color.color(0.3421f, 0.33893f, 0.33893f)));
            categoryButtonsVBox.getChildren().add(button);
        }
    }

    private void loadDrawingsByCategory(String category) {
        List<Draw> drawings = PersistenceManager.getDrawingsByCategory(entityManager, category);

        drawingsGridPane.getChildren().clear();
        int row = 0;
        int column = 0;

        for (Draw draw : drawings) {
            // Here you should convert draw.getData() (byte[]) to an ImageView
            // Assuming there's a utility method byteArrayToImageView for this
            ImageView imageView = byteArrayToImageView(draw.getData());
            imageView.setOnMouseClicked(event -> showDrawingDetails(draw));
            drawingsGridPane.add(imageView, column, row);

            column++;
            if (column == 4) {
                column = 0;
                row++;
            }
        }
    }

    private ImageView byteArrayToImageView(byte[] data) {
        Image image = new Image(new ByteArrayInputStream(data));
        ImageView imageView = new ImageView(image);
        imageView.setFitWidth(100);
        imageView.setFitHeight(100);
        return imageView;
    }

    private void showDrawingDetails(Draw draw) {
        try {
            FXMLLoader loader = new FXMLLoader(getClass().getResource("fxml/drawing-details-view.fxml"));
            Stage stage = new Stage();
            stage.setScene(new Scene(loader.load()));
            DrawingDetailsController controller = loader.getController();
            controller.setEntityManager(entityManager);
            controller.setDrawing(draw);
            stage.show();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @FXML
    private void handleBackButtonAction(ActionEvent event) throws IOException {
        Stage stage = (Stage) backButton.getScene().getWindow();
        FXMLLoader loader = new FXMLLoader(getClass().getResource("fxml/menu-view.fxml"));
        Parent root = loader.load();
        Scene scene = new Scene(root);
        stage.setScene(scene);
        stage.show();
    }
}
