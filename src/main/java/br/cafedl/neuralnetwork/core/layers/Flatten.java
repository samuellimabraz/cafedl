package br.cafedl.neuralnetwork.core.layers;

import dev.morphia.annotations.Entity;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

@Entity("flatten")
public class Flatten extends Layer<Flatten> {
    public Flatten() {
    }

    @Override
    public INDArray forward(INDArray inputs) {
        this.setInput(inputs);
        return inputs.reshape(new int[] {(int) inputs.size(0), -1 });
    }

    @Override
    public INDArray backward(INDArray gradout) {
        return gradout.reshape(this.getInput().shape());
    }

    @Override
    public String toString() {
        return "Flatten";
    }

    /**
     * @param dis
     * @return
     * @throws IOException
     */
    @Override
    public Flatten loadAdditional(DataInputStream dis) throws IOException {
        return this;
    }

    /**
     * @param dos
     * @throws IOException
     */
    @Override
    public void saveAdditional(DataOutputStream dos) throws IOException {

    }
}
