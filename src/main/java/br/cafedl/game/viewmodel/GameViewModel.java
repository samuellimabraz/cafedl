package br.cafedl.game.viewmodel;

import br.cafedl.game.model.Draw;
import br.cafedl.game.model.GameSession;
import br.cafedl.game.model.PredictionResult;
import br.cafedl.game.controller.GameController;
import br.cafedl.game.model.database.PersistenceManager;
import javafx.animation.PauseTransition;
import javafx.application.Platform;
import javafx.scene.image.*;
import javafx.embed.swing.SwingFXUtils;
import javafx.util.Duration;

import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;

import javax.imageio.ImageIO;
import javax.persistence.EntityManager;

public class GameViewModel extends ViewModel {
    private final GameController controller;
    private final EntityManager entityManager;

    public GameViewModel(GameController controller, GameSession session, EntityManager entityManager) {
        super(session);
        session.setViewModel(this);
        this.controller = controller;
        this.entityManager = entityManager;
    }


    public void nextRound(boolean correct) throws IOException {
        // Persist the current round before going to the next one
        if (correct) {
            if (entityManager != null) {
                PersistenceManager.persistAll(entityManager, Arrays.asList(
                        session.getCurrentRound().getDrawing().getPredictionResult(),
                        session.getCurrentRound().getDrawing(),
                        session.getCurrentRound())
                );
            }
        }

        if (session.nextRound())
            controller.nextRound();
        else {
            endSessionGame();
            endControllerGame();
        }
    }

    public void endSessionGame() {
        session.endGame();
        if (entityManager != null) {
            PersistenceManager.updateEndTime(entityManager, session.getId(), session.getEndTime());
        }
    }

    public void endControllerGame() throws IOException {
        controller.endGame();
    }

    public void sendDrawing(WritableImage drawing) throws IOException {
        controller.disableCanvas(true);
        session.getCurrentRound().pause();

        Draw draw = new Draw();
        session.setDrawing(draw);

        // Convert the WritableImage to a byte[]
        BufferedImage bufferedImage = SwingFXUtils.fromFXImage(drawing, null);
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        ImageIO.write(bufferedImage, "png", byteArrayOutputStream);
        byte[] byteArray = byteArrayOutputStream.toByteArray();

        draw.setData(byteArray);

        //displayImage(arrayToImage(imgArray.dup()));
        List<PredictionResult> predictions = session.predict();

        // Ordenar as predições em ordem decrescente de probabilidade
        predictions.sort((p1, p2) -> Double.compare(p2.getProbability(), p1.getProbability()));

        Optional<PredictionResult> matchedPrediction = predictions.stream()
                .filter(prediction -> prediction.getCategory().equals(session.getCurrentRound().getCategory()))
                .findFirst();

        matchedPrediction.ifPresent(draw::setPredictionResult);

        PredictionResult result = predictions.get(0);
        String category = result.getCategory();
        double prob = result.getProbability();

        System.out.println("Predition result: " + category + " - " + prob);
        controller.updatePredictions(predictions);

        // Verificar se a predição está correta
        if (category.equals(session.getCurrentRound().getCategory()) && prob > 50.0f) {
            System.out.println("Correct!");
            // Enviar a imagem para a simulação do banco de dados

            PauseTransition pause = new PauseTransition(Duration.seconds(2.0));
            pause.setOnFinished(event -> {
                try {
                    nextRound(true);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            });
            pause.play();
        } else {
            session.getCurrentRound().resume(); // Resume the timer
            controller.disableCanvas(false);
        }
    }


    public void updateTimer(int time) {
        if (time == 0) {
            System.out.println("Time is up!");
            Platform.runLater(() -> {
                try {
                    nextRound(false);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            });
        } else {
            Platform.runLater(() -> controller.updateTime(time));
        }
    }

    @Override
    public void updateView() {
        controller.updateCategory(session.getCurrentRound().getCategory());
    }

}
