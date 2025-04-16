package br.cafedl.neuralnetwork.core.layers;

import dev.morphia.annotations.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

@Entity
public abstract class TrainableLayer extends Layer<TrainableLayer> {
    @Transient
    protected INDArray params;
    @Transient
    protected INDArray grads;
    @Property
    public boolean trainable = true;

    @Property("params")
    private byte[] paramsData;
    @Property("grads")
    private byte[] gradsData;

    @PrePersist
    public void updateDataArrays() throws IOException {
        if (params != null) {
            paramsData = Nd4j.toByteArray(params);
        }
        if (grads != null) {
            gradsData = Nd4j.toByteArray(grads);
        }
    }

    @PostLoad
    public void loadDataArrays() {
        if (paramsData != null) {
            params = Nd4j.fromByteArray(paramsData);
        }
        if (gradsData != null) {
            grads = Nd4j.fromByteArray(gradsData);
        }
    }

    public void setup(INDArray input) { this.input = input; }

    public INDArray getParams() { return params; }

    public void setParams(INDArray params) {
        this.params = params;
    }

    public INDArray getGrads() { return grads; }

    public void setGrads(INDArray grads) { this.grads = grads; }

    public void setTrainable(boolean trainable) { this.trainable = trainable; }

    public boolean isTrainable() { return trainable; }

    @Override
    public void saveAdditional(DataOutputStream dos) throws IOException {
        if (input == null || output == null) {
            throw new IOException("Layer is not initialized");
        }
        Nd4j.write(input, dos);
        Nd4j.write(output, dos);
        Nd4j.write((params == null) ? Nd4j.zeros(1) : params , dos);
        Nd4j.write((grads == null) ? Nd4j.zeros(1): grads, dos);
    }

    @Override
    public TrainableLayer loadAdditional(DataInputStream dis) throws IOException {
        input = Nd4j.read(dis);
        output = Nd4j.read(dis);
        params = Nd4j.read(dis);
        grads = Nd4j.read(dis);
        return this;
    }
}
