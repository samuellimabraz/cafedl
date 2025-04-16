package br.cafedl.neuralnetwork.core.layers;


import dev.morphia.annotations.Entity;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.stream.IntStream;

@Entity
public class MaxPooling2D extends Layer<MaxPooling2D> {
    private int poolSize;
    private int stride;

    public MaxPooling2D(int poolSize, int stride) {
        this.poolSize = poolSize;
        this.stride = stride;
    }

    public MaxPooling2D() {
        this.poolSize = 2;
        this.stride = 2;
    }


    public INDArray forward(INDArray inputs) {
        this.setInput(inputs);
        int m = (int) inputs.size(0);
        int n_H_prev = (int) inputs.size(1);
        int n_W_prev = (int) inputs.size(2);
        int n_C_prev = (int) inputs.size(3);

        int n_H = 1 + (n_H_prev - poolSize) / stride;
        int n_W = 1 + (n_W_prev - poolSize) / stride;
        int n_C = n_C_prev;

        // Pre-calculate indices
        int[] vert_starts = IntStream.range(0, n_H).map(h -> h * stride).toArray();
        int[] horiz_starts = IntStream.range(0, n_W).map(w -> w * stride).toArray();

        INDArray output = Nd4j.zeros(DataType.DOUBLE, m, n_H, n_W, n_C);


        for (int i = 0; i < m; i++) {
            for (int h = 0; h < n_H; h++) {
                int vert_end = vert_starts[h] + poolSize;

                for (int w = 0; w < n_W; w++) {
                    int horiz_end = horiz_starts[w] + poolSize;

                    for (int c = 0; c < n_C; c++) {
                        INDArray a_prev_slice = inputs.get(NDArrayIndex.point(i), NDArrayIndex.interval(vert_starts[h], vert_end), NDArrayIndex.interval(horiz_starts[w], horiz_end), NDArrayIndex.point(c));
                        output.put(new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.point(h), NDArrayIndex.point(w), NDArrayIndex.point(c)}, a_prev_slice.maxNumber());
                    }
                }
            }
        }

        this.setOutput(output);
        return output;
    }

    @Override
    public INDArray backward(INDArray gradout) {
        INDArray APrev = this.getInput();
        //int m = (int) APrev.size(0);
        int n_H_prev = (int) APrev.size(1);
        int n_W_prev = (int) APrev.size(2);
        int n_C_prev = (int) APrev.size(3);

        int m = (int) gradout.size(0);
        int n_H = (int) gradout.size(1);
        int n_W = (int) gradout.size(2);
        int n_C = (int) gradout.size(3);

        INDArray dAPrev = Nd4j.zeros(DataType.DOUBLE, m, n_H_prev, n_W_prev, n_C_prev);

        for (int i = 0; i < m; i++) {
            INDArray aPrev = APrev.get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());

            for (int h = 0; h < n_H; h++) {
                int vert_start = h * stride;
                int vert_end = vert_start + poolSize;

                for (int w = 0; w < n_W; w++) {
                    int horiz_start = w * stride;
                    int horiz_end = horiz_start + poolSize;

                    for (int c = 0; c < n_C; c++) {
                        INDArray aPrevSlice = aPrev.get(NDArrayIndex.interval(vert_start, vert_end), NDArrayIndex.interval(horiz_start, horiz_end), NDArrayIndex.point(c));
                        double max = aPrevSlice.maxNumber().doubleValue();
                        INDArray mask = aPrevSlice.eq(max).castTo(DataType.DOUBLE);
                        dAPrev.put(new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.interval(vert_start, vert_end), NDArrayIndex.interval(horiz_start, horiz_end), NDArrayIndex.point(c)}, mask.mul(gradout.getDouble(i, h, w, c)));
                    }
                }
            }
        }

        return dAPrev;
    }


    @Override
    public String toString() {
        return "MaxPooling2D(poolSize=" + poolSize + ", stride=" + stride + ")";
    }

    /**
     * @param dis 
     * @return
     * @throws IOException
     */
    @Override
    public MaxPooling2D loadAdditional(DataInputStream dis) throws IOException {
        this.poolSize = dis.readInt();
        this.stride = dis.readInt();
        return this;
    }

    /**
     * @param dos 
     * @throws IOException
     */
    @Override
    public void saveAdditional(DataOutputStream dos) throws IOException {
        dos.writeInt(poolSize);
        dos.writeInt(stride);
    }
}
