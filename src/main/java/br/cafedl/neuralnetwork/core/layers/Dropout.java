package br.cafedl.neuralnetwork.core.layers;

import dev.morphia.annotations.Entity;
import dev.morphia.annotations.Property;
import dev.morphia.annotations.Transient;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

@Entity
public class Dropout extends Layer<Dropout> {
    @Property
    private double dropoutRate = 0.0;
    @Transient
    private INDArray mask;

    public Dropout(double dropoutRate) {
        this.dropoutRate = dropoutRate;
    }

    public Dropout() {
        this(0.0);
    }

    @Override
    public INDArray forward(INDArray inputs) {
        this.setInput(inputs);
        if (this.inference) {
            this.setOutput(inputs.mul(1.0 - dropoutRate));
        } else {
            if (dropoutRate > 0.0) {
                this.mask = Nd4j.rand(DataType.DOUBLE, input.shape()).gt(dropoutRate);
                this.setOutput(input.mul(this.mask));
            } else {
                this.setOutput(input);
            }
        }

        return this.getOutput();
    }

    @Override
    public INDArray backward(INDArray gradout) {
        if (dropoutRate > 0.0) {
            return gradout.mul(this.mask);
        } else {
            return gradout;
        }
    }

    @Override
    public String toString() {
        return "Dropout(" + this.dropoutRate + ")";
    }

    public double getDropoutRate() {
        return this.dropoutRate;
    }

    public void setDropoutRate(double dropoutRate) {
        this.dropoutRate = dropoutRate;
    }

    public INDArray getMask() {
        return this.mask;
    }

    public void setMask(INDArray mask) {
        this.mask = mask;
    }

    /**
     * @param dos
     * @throws IOException
     */
    @Override
    public void saveAdditional(DataOutputStream dos) throws IOException {
        dos.writeDouble(this.dropoutRate);
    }


    /**
     * @param dis
     * @return
     * @throws IOException
     */
    @Override
    public Dropout loadAdditional(DataInputStream dis) throws IOException {
        dropoutRate = dis.readDouble();
        return this;
    }

}
