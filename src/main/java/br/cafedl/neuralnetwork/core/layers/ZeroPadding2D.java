package br.cafedl.neuralnetwork.core.layers;

import dev.morphia.annotations.Entity;
import dev.morphia.annotations.Property;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

@Entity
public class ZeroPadding2D extends Layer<ZeroPadding2D> {
    @Property
    protected int padding;

    public ZeroPadding2D(int padding) {
        this.padding = padding;
    }

    public ZeroPadding2D() {
        this.padding = 1;
    }

    @Override
    public INDArray forward(INDArray inputs) {
        this.setInput(inputs.dup());
        INDArray output = Nd4j.pad(inputs, new int[][]{{0, 0}, {padding, padding}, {padding, padding}, {0, 0}});
        this.setOutput(output);
        return output;
    }

    @Override
    public INDArray backward(INDArray gradout) {
        return gradout.get(
                NDArrayIndex.all(),
                NDArrayIndex.interval(padding, gradout.shape()[1] - padding),
                NDArrayIndex.interval(padding, gradout.shape()[2] - padding),
                NDArrayIndex.all()
        );
    }

    @Override
    public ZeroPadding2D loadAdditional(DataInputStream dis) throws IOException {
        return new ZeroPadding2D(dis.readInt());
    }

    @Override
    public void saveAdditional(DataOutputStream dos) throws IOException {
        dos.writeInt(padding);
    }

    @Override
    public String toString(){
        return "ZeroPadding2D(padding=" + this.padding + ")";
    }
}
