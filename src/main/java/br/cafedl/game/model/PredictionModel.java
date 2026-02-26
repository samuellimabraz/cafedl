package br.cafedl.game.model;

import br.cafedl.game.model.database.StringListConverter;

import jakarta.persistence.*;
import java.util.List;

@Entity
@Inheritance(strategy = InheritanceType.SINGLE_TABLE)
public abstract class PredictionModel {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "database_type")
    public String databaseType;
    @Column(name = "model_name")
    public String modelName;
    @Transient
    protected boolean modelLoaded = false;

    @Convert(converter = StringListConverter.class)
    protected List<String> categories;

    public abstract void loadModel();

    public abstract List<PredictionResult> predict(byte[] data);

    public List<String> getCategories() {
        return categories;
    }
}
