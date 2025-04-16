package br.cafedl.neuralnetwork.core.models;

import br.cafedl.neuralnetwork.core.layers.Layer;

import java.util.ArrayList;
import java.util.List;

public class ModelBuilder {
    public List<Layer> layers = new ArrayList<>();

    public ModelBuilder add(Layer layer) {
        this.layers.add(layer);
        if (this.layers.size() > 1) {
            int i = this.layers.size() - 2;
            this.layers.get(i).nextLayer = layer;
            System.out.println("Connect: " + layers.get(i).getName() + " -> " + layers.get(i + 1).getName());
        }
        return this;
    }
    public NeuralNetwork build() {
        return new NeuralNetwork(this);
    }

}

