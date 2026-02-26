package br.cafedl.game.model;

import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;

@Entity
public class PredictionResult {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private final String category;
    private final double probability;

    public PredictionResult() {
        this.category = null;
        this.probability = 0;
    }

    public PredictionResult(String category, double probability) {
        this.category = category;
        this.probability = probability;
    }

    public String getCategory() {
        return category;
    }

    public double getProbability() {
        return probability;
    }
}
