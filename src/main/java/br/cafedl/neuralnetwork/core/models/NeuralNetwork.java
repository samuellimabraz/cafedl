package br.cafedl.neuralnetwork.core.models;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

import br.cafedl.neuralnetwork.core.layers.LayerLoader;
import br.cafedl.neuralnetwork.core.layers.TrainableLayer;
import br.cafedl.neuralnetwork.core.layers.Layer;
import dev.morphia.annotations.*;
import org.bson.types.ObjectId;
import org.nd4j.linalg.api.ndarray.INDArray;



@Entity
public class NeuralNetwork {
    @Id
    public ObjectId id = new ObjectId();

    @Property
    public String name = "neural_network_" + UUID.randomUUID().toString();

    @Reference(lazy = true)
    protected List<Layer> layers = new ArrayList<>();

    @Transient
    private List<TrainableLayer> trainableLayers = new ArrayList<>();

    @Transient
    private INDArray output = null;

    public NeuralNetwork(ModelBuilder modelBuilder) {
        this.setLayers(modelBuilder.layers);

        this.trainableLayers = layers.stream()
                .filter(layer -> layer instanceof TrainableLayer)
                .map(layer -> (TrainableLayer) layer)
                .collect(Collectors.toList());
    }

    protected NeuralNetwork() {
    }

    @PostLoad
    public void initTrainableLayers() {
        this.trainableLayers = layers.stream()
                .filter(layer -> layer instanceof TrainableLayer)
                .map(layer -> (TrainableLayer) layer)
                .collect(Collectors.toList());
    }

    public ObjectId getId() {
        return id;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getName() {
        return name;
    }

    public INDArray predict(INDArray x) {
        output = x;
        for (Layer layer : layers)
            output = layer.forward(output);
        return output;
    }
    public void backPropagation(INDArray gradout) {
        for (int i = layers.size() - 1; i >= 0; i--)
            gradout = layers.get(i).backward(gradout);
    }

    public List<Layer> getLayers() {
        return layers;
    }

    public List<TrainableLayer> getTrainableLayers() {
        if (trainableLayers.isEmpty()) {
            trainableLayers = layers.stream()
                    .filter(layer -> layer instanceof TrainableLayer)
                    .map(layer -> (TrainableLayer) layer)
                    .collect(Collectors.toList());
        }
        return trainableLayers;
    }

    private void setLayers(List<Layer> layers) {
        this.layers = layers;
    }

    public void setTrainable(boolean trainable) {
        trainableLayers.forEach(layer -> layer.setTrainable(trainable));
    }

    public void setInference(boolean inference) {
        layers.forEach(layer -> layer.setInference(inference));
    }

    public void saveModel(String filePath) throws IOException {
        DataOutputStream dos = new DataOutputStream(Files.newOutputStream(Paths.get(filePath)));

        dos.writeUTF(name);
        // Salva o número de camadas
        dos.writeInt(layers.size());

        // Salva cada camada
        layers.forEach(layer -> {
            try {
                layer.save(dos);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        });

        dos.close();

        System.out.println("Model saved successfully");
    }

    public static NeuralNetwork loadModel(String filePath) throws Exception {
        DataInputStream dis = new DataInputStream(new FileInputStream(filePath));

        // Carrega o nome
        String name = dis.readUTF();

        // Carrega o número de camadas
        int numLayers = dis.readInt();

        // Carrega cada camada
        ModelBuilder modelBuilder = new ModelBuilder();
        for (int i = 0; i < numLayers; i++) {
            Layer<?> layer = LayerLoader.load(dis);
            modelBuilder.add(layer);
        }

        dis.close();

        System.out.println("Model loaded successfully");

        NeuralNetwork model =  modelBuilder.build();
        model.setName(name);

        return model;
    }

    public String toString() {
        return "NeuralNetwork{" +
                "name='" + name + '\'' +
                ", layers=" + layers +
                '}';
    }
}