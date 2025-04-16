package br.cafedl.game.model;

import javax.persistence.*;

@Entity
public class Draw {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Lob
    private byte[] data;

    @OneToOne
    private PredictionResult predictionResult;

    public Draw() {
    }

    public Draw(byte[] data, PredictionResult predictionResult) {
        this.data = data;
        this.predictionResult = predictionResult;
    }

    public Draw(byte[] data) {
        this.data = data;
    }

    public byte[] getData() {
        return data;
    }

    public void setData(byte[] data) {
        this.data = data;
    }

    public String getCategory() {
        return predictionResult.getCategory();
    }

    public double getConfidence() {
        return predictionResult.getProbability();
    }

    public void setPredictionResult(PredictionResult predictionResult) {
        this.predictionResult = predictionResult;
    }

    public PredictionResult getPredictionResult() {
        return predictionResult;
    }

    public Long getId() {
        return id;
    }

}
