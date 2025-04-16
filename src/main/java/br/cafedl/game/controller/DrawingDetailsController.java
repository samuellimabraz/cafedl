package br.cafedl.game.controller;

import br.cafedl.game.model.Draw;
import br.cafedl.game.model.Round;
import br.cafedl.game.model.database.PersistenceManager;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;

import javax.persistence.EntityManager;
import java.io.ByteArrayInputStream;

public class DrawingDetailsController {
    @FXML
    public Label drawID;

    @FXML
    private ImageView drawingImageView;

    @FXML
    private Label predictionCategoryLabel;

    @FXML
    private Label predictionProbabilityLabel;

    @FXML
    private Label roundIdLabel;

    @FXML
    private Button deleteButton;

    private Draw draw;
    private EntityManager entityManager;

    public void setEntityManager(EntityManager entityManager) {
        this.entityManager = entityManager;
    }

    public void setDrawing(Draw draw) {
        this.draw = draw;

        Image image = new Image(new ByteArrayInputStream(draw.getData()));
        drawingImageView.setImage(image);

        drawID.setText("ID: " + draw.getId());
        predictionCategoryLabel.setText("Category: " + draw.getPredictionResult().getCategory());
        predictionProbabilityLabel.setText("Probability: " + String.valueOf(draw.getPredictionResult().getProbability()));

        Round round = PersistenceManager.getRoundByDrawId(entityManager, draw.getId());
        roundIdLabel.setText("Round ID: " + round.getId());

        deleteButton.setOnAction(event -> deleteDrawing());
    }

    private void deleteDrawing() {
        PersistenceManager.deleteDraw(entityManager, draw.getId());
        deleteButton.getScene().getWindow().hide();
    }
}
