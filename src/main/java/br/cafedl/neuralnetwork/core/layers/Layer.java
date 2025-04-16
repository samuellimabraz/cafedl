package br.cafedl.neuralnetwork.core.layers;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.UUID;

import dev.morphia.annotations.*;
import dev.morphia.Datastore;
import org.bson.types.ObjectId;
import org.nd4j.linalg.api.ndarray.INDArray;

@Entity
public abstract class Layer <T extends Layer<T>> {
    @Id
    public ObjectId id = new ObjectId();

    @Reference(ignoreMissing = true, lazy = true)
    public Layer<?> nextLayer = null;

    @Property
    public String name = "layer_" + this.getClass().getSimpleName() + "_" + UUID.randomUUID();

    @Transient
    protected INDArray input;
    @Transient
    protected INDArray output;
    @Property
    public boolean inference = false;

    public abstract INDArray forward(INDArray inputs);
    public abstract INDArray backward(INDArray gradout);

    public String toString() {
        return this.getClass().getSimpleName();
    }

    public T load(DataInputStream dis) throws IOException {
        return loadAdditional(dis);
    }

    public abstract T loadAdditional(DataInputStream dis) throws IOException;

    public void save(DataOutputStream dos) throws IOException {
        dos.writeUTF(this.getClass().getSimpleName());
        this.saveAdditional(dos);
    }
    public abstract void saveAdditional(DataOutputStream dos) throws IOException;

    public void save(Datastore datastore) {
        System.out.println("Saving " + this.getClass().getSimpleName());
        datastore.save(this);
    }

    public INDArray getInput() {
        return input;
    }
    public INDArray getOutput() {
        return output;
    }

    public void setInput(INDArray input) {
        this.input = input;
    }
    public void setOutput(INDArray output) {
        this.output = output;
    }

    public void setInference(boolean inference) {
        this.inference = inference;
    }

    public boolean isInference() {
        return inference;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public ObjectId getId() {
        return id;
    }
}
