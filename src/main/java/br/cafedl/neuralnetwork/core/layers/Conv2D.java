package br.cafedl.neuralnetwork.core.layers;

import br.cafedl.neuralnetwork.core.activation.*;
import dev.morphia.Datastore;
import dev.morphia.annotations.*;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

@Entity
public class Conv2D extends TrainableLayer {
    protected int filters;
    protected int kernelSize;
    protected List<Integer> strides;
    protected String padding;
    @Transient
    protected IActivation activation;
    @Property
    public String activationType;
    protected String kernelInitializer;
    @Transient
    protected int pad;

    protected Layer<ZeroPadding2D> zeroPadding2D;

    @Transient
    protected int m ,nHInput, nWInput, nCInput;
    @Transient
    protected int nHOutput, nWOutput, nCOutput;

    @Transient
    private boolean isInitialized = false;

    @Transient
    private INDArray paddedInputs;
    @Transient
    private INDArray weightsC, biasesC, aPrev, aSlicePrev;
    @Transient
    private int[] vert_starts, horiz_starts;

    @PostLoad
    public void loadActivationType() {
        if (activationType != null) {
            activation = Activation.create(activationType);
        }
        this.isInitialized = false;
    }

    public Conv2D(int filters, int kernelSize, List<Integer> strides, String padding, IActivation activation, String kernelInitializer) {
        this.filters = filters;
        this.kernelSize = kernelSize;
        this.strides = strides;
        this.padding = padding;
        this.activation = activation;
        this.kernelInitializer = kernelInitializer;
        this.activationType = activation.getClass().getSimpleName().toLowerCase();
        this.initZeroPadding2D();
    }

    public  Conv2D(int filters, int kernelSize, String padding, IActivation activation,  String kernelInitializer) {
        this(filters, kernelSize, Arrays.asList(1, 1), padding, activation, kernelInitializer);
    }

    public Conv2D(int filters, int kernelSize, IActivation activation) {
        this(filters, kernelSize, "valid", activation, "random");
    }

    public Conv2D(int filters, int kernelSize) {
        this(filters, kernelSize, "valid", Activation.create("linear"), "random");
    }

    public Conv2D() {
        this(1, 3, "valid", Activation.create("linear"), "random");
    }

    private void initZeroPadding2D() {
        if (padding.equalsIgnoreCase("same")) {
            this.pad = kernelSize / 2;
        } else if (padding.equalsIgnoreCase("valid")) {
            this.pad = 0;
        } else {
            throw new IllegalArgumentException("Invalid padding: " + padding);
        }
        this.zeroPadding2D = new ZeroPadding2D(pad);
    }

    @Override
    public void setup(INDArray inputs) {
        // Implement the setup method for a convolution function
        long[] shape = inputs.shape();
        this.m = (int) shape[0];
        this.nHInput = (int) shape[1];
        this.nWInput = (int) shape[2];
        this.nCInput = (int) shape[3];

        this.nHOutput = (((nHInput - kernelSize + 2 * pad) / strides.get(0)) + 1); // int((nHInput - kernelSize + 2 * pad) / strides[0]) + 1
        this.nWOutput = (((nWInput - kernelSize + 2 * pad) / strides.get(1)) + 1); // int((nWInput - kernelSize + 2 * pad) / strides[1]) + 1
        this.nCOutput = filters;

        this.vert_starts = IntStream.range(0, nHOutput).map(h -> h * strides.get(0)).toArray();
        this.horiz_starts = IntStream.range(0, nWOutput).map(w -> w * strides.get(1)).toArray();

        if (params == null || grads == null) {
            System.out.println("Initializing weights and biases");
            double scale = getScale();

            // Weights and bias represented as a single 5D tensor, where the last dimension is used for the bias
            // The tensor has shape (kernelSize, kernelSize, nc, filters, 2)
            params = Nd4j.create(DataType.DOUBLE, kernelSize, kernelSize, nCInput, filters, 2);

            INDArray W = Nd4j.randn(DataType.DOUBLE, kernelSize, kernelSize, nCInput, filters).mul(scale);
            INDArray b = Nd4j.zeros(DataType.DOUBLE, 1, 1, 1, filters); // Initialize b with shape (1, 1, 1, filters)

            // Broadcast b to have the same shape as W
            b = b.broadcast(kernelSize, kernelSize, nCInput, filters);

            this.setWeights(W);
            this.setBiases(b);

            // Gradients of weights and bias
            grads = Nd4j.zeros(DataType.DOUBLE, params.shape());
        }
        this.isInitialized = true;
    }

    private double getScale() {
        double scale;
        int fanIn = kernelSize * kernelSize * nCInput;
        int fanOut = kernelSize * kernelSize * filters;

        if (this.kernelInitializer.equalsIgnoreCase("xavier")) {
            if (this.activation instanceof ReLU || this.activation instanceof LeakyReLU || this.activation instanceof SiLU) {
                scale = Math.sqrt(2.0 / (fanIn + fanOut)); // Xavier initialization for ReLU variants
            } else if (this.activation instanceof TanH || this.activation instanceof Sigmoid || this.activation instanceof Softmax || this.activation instanceof Linear) {
                scale = Math.sqrt(1.0 / (fanIn + fanOut)); // Xavier initialization for other activations
            } else {
                throw new IllegalArgumentException("Xavier initialization not supported for this activation function");
            }
        } else if (this.kernelInitializer.equalsIgnoreCase("he")) {
            // He initialization for ReLU variants
            if (this.activation instanceof ReLU || this.activation instanceof LeakyReLU || this.activation instanceof SiLU || this.activation instanceof Linear) {
                // For ReLU, LeakyReLU, and SiLU
                scale = Math.sqrt(2.0 / fanIn);
            } else {
                throw new IllegalArgumentException("He initialization is generally used for ReLU, LeakyReLU, and SiLU activations");
            }
        } else if (this.kernelInitializer.equalsIgnoreCase("zero")) {
            scale = 0.0;
        } else {
            scale = 0.01;
        }
        return scale;
    }

    public INDArray forward2(INDArray inputs) {
        if (!this.isInitialized) {
            this.setup(inputs);
        }
        setInput(inputs);

        this.m = (int) inputs.shape()[0]; // Batch size
        // Implement the forward propagation for a convolution function
        output = Nd4j.zeros(DataType.DOUBLE, m, nHOutput, nWOutput, filters);

        // Apply padding to the input volume
        this.paddedInputs = zeroPadding2D.forward(inputs);


        // Loop over the batch of training examples
        for (int i = 0; i < m; i++) {
            aPrev = this.paddedInputs.get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
            // Loop over vertical axis of the output volume
            for (int h = 0; h < nHOutput; h++) {
                // Find the vertical start and end of the current "slice"
                int vert_start = h * strides.get(0);
                int vert_end = vert_start + kernelSize;

                // Loop over horizontal axis of the output volume
                for (int w = 0; w < nWOutput; w++) {
                    // Find the horizontal start and end of the current "slice"
                    int horiz_start = w * strides.get(1);
                    int horiz_end = horiz_start + kernelSize;

                    // Loop over channels (= #filters) of the output volume
                    for (int c = 0; c < filters; c++) {
                        // Use the corners to define the (3D) slice of inputs
                        aSlicePrev = aPrev.get(NDArrayIndex.interval(vert_start, vert_end), NDArrayIndex.interval(horiz_start, horiz_end), NDArrayIndex.all());

                        // Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron
                        weightsC = params.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(c), NDArrayIndex.point(0));
                        biasesC = params.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(c), NDArrayIndex.point(1));

                        output.putScalar(new int[]{i, h, w, c}, aSlicePrev.mul(weightsC).sumNumber().doubleValue() + biasesC.getDouble(0));
                    }
                }
            }
        }

        output = activation.forward(output);

        return output;
    }

    /**
     * Implements the forward propagation for a convolution function.
     *
     * @param inputs input data (batch size, height, width, channels)
     *               <p>batch size: number of training examples
     *               <p>height: height of the input volume
     *               <p>width: width of the input volume
     *               <p>channels: number of channels of the input volume
     *               (INDArray).
     *
     * @return output of the convolutional layer (INDArray).
     */
    @Override
    public INDArray forward(INDArray inputs) {
        if (!this.isInitialized) {
            this.setup(inputs);
        }
        setInput(inputs);

        this.m = (int) inputs.shape()[0]; // Batch size

        // Apply padding to the input volume
        this.paddedInputs = zeroPadding2D.forward(inputs);

        // Reshape the weights for matrix multiplication
        INDArray weightsReshaped = params.
                get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(0)).
                permute(3, 2, 0, 1).
                reshape(filters, (long) kernelSize * kernelSize * nCInput);

        // Initialize output tensor
        output = Nd4j.zeros(DataType.DOUBLE, m, nHOutput, nWOutput, filters);

        INDArray image, patches, convResult;

        // Extract image patches and perform convolution
        for (int i = 0; i < m; i++) {
            image = this.paddedInputs.get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());

            // Extract patches
            patches = Nd4j.zeros(DataType.DOUBLE, nHOutput, nWOutput, kernelSize, kernelSize, nCInput);
            for (int h = 0; h < nHOutput; h++) {
                for (int w = 0; w < nWOutput; w++) {
                    INDArray patch = image.get(NDArrayIndex.interval(vert_starts[h], vert_starts[h] + kernelSize), NDArrayIndex.interval(horiz_starts[w], horiz_starts[w] + kernelSize), NDArrayIndex.all());
                    patches.put(new INDArrayIndex[]{NDArrayIndex.point(h), NDArrayIndex.point(w), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all()}, patch);
                }
            }

            // Perform convolution
            convResult = patches.reshape((long) nHOutput * nWOutput, (long) kernelSize * kernelSize * nCInput).mmul(weightsReshaped.transpose());

            // Add bias and apply activation function
            convResult = convResult.addiRowVector(params.get(NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(1))).
                    reshape(nHOutput, nWOutput, filters);

            output.put(new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all()}, activation.forward(convResult));
        }

        return output;
    }


    /**
     * Implements the backward propagation for a convolution function.
     *
     * @param gradout gradient of the cost with respect to the output of the previous layer (INDArray).
     * @return gradient of the cost with respect to the input of the convolutional layer (INDArray).
     */
    @Override
    public INDArray backward(INDArray gradout) {
        // Initialize dA_prev, dW, db with the correct shapes
        INDArray dA_prev = Nd4j.zeros(DataType.DOUBLE, m, nHInput, nWInput, nCInput);
        INDArray dW = Nd4j.zeros(DataType.DOUBLE, getWeights().shape());
        INDArray db = Nd4j.zeros(DataType.DOUBLE, getBiases().shape());

        long[] dZshape = gradout.shape();
        long m = dZshape[0];
        int endIdxH = (int) (dA_prev.shape()[1] - pad);
        int endIdxW = (int) (dA_prev.shape()[2] - pad);

        // Apply activation backward to gradout
        gradout.muli(activation.backward(output));

        // Pad inputs and dA_prev
        INDArray dA_prev_pad = zeroPadding2D.forward(dA_prev);

        // Reshape the weights for matrix multiplication
        INDArray weightsReshaped = this.getWeights().permute(3, 2, 0, 1).reshape(filters, kernelSize * kernelSize * nCInput);

        // Initialize temporary arrays
        INDArray a_prev_pad, da_prev_pad, patches, dZ_reshaped, dW_temp, db_temp, dA_prev_temp;

        for (int i = 0; i < m; i++) {
            a_prev_pad = paddedInputs.get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
            da_prev_pad = dA_prev_pad.get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());

            // Extract patches
            patches = Nd4j.zeros(DataType.DOUBLE, nHOutput, nWOutput, kernelSize, kernelSize, nCInput);
            for (int h = 0; h < nHOutput; h++) {
                for (int w = 0; w < nWOutput; w++) {
                    INDArray patch = a_prev_pad.get(NDArrayIndex.interval(vert_starts[h], vert_starts[h] + kernelSize), NDArrayIndex.interval(horiz_starts[w], horiz_starts[w] + kernelSize), NDArrayIndex.all());
                    patches.put(new INDArrayIndex[]{NDArrayIndex.point(h), NDArrayIndex.point(w), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all()}, patch);
                }
            }

            // Perform convolution
            dZ_reshaped = gradout.get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all()).reshape(nHOutput * nWOutput, filters);
            dW_temp = patches.reshape(nHOutput * nWOutput, kernelSize * kernelSize * nCInput).transpose().mmul(dZ_reshaped);
            dW.addi(dW_temp.reshape(kernelSize, kernelSize, nCInput, filters));

            // Compute db
            db_temp = dZ_reshaped.sum(0);
            db.addi(db_temp.reshape(1, 1, 1, filters));

            // Calculate dA_prev
            dA_prev_temp = dZ_reshaped.mmul(weightsReshaped).reshape(nHOutput, nWOutput, kernelSize, kernelSize, nCInput);
            for (int h = 0; h < nHOutput; h++) {
                for (int w = 0; w < nWOutput; w++) {
                    da_prev_pad.get(NDArrayIndex.interval(vert_starts[h], vert_starts[h] + kernelSize), NDArrayIndex.interval(horiz_starts[w], horiz_starts[w] + kernelSize), NDArrayIndex.all()).addi(dA_prev_temp.get(NDArrayIndex.point(h), NDArrayIndex.point(w), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all()));
                }
            }

            dA_prev.put(new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.interval(pad, endIdxH), NDArrayIndex.interval(pad, endIdxW), NDArrayIndex.all()},
                    da_prev_pad.get(NDArrayIndex.interval(pad, endIdxH), NDArrayIndex.interval(pad, endIdxW), NDArrayIndex.all()));
        }

        // Update gradients in params
        grads.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(0)).assign(dW);
        grads.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(1)).assign(db);

        return dA_prev;
    }

    public INDArray getWeights() {
        return params.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(0));
    }

    public INDArray getBiases() {
        return params.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(1));
    }

    public void setWeights(INDArray weights) {
        params.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(0)).assign(weights);
    }

    public void setBiases(INDArray biases) {
        params.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(1)).assign(biases);
    }

    public INDArray getGradWeights() {
        return grads.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(0));
    }

    public INDArray getGradBiases() {
        return grads.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(1));
    }

    public int getFilters() {
        return filters;
    }

    public int getKernelSize() {
        return kernelSize;
    }

    public List<Integer>  getStrides() {
        return strides;
    }

    public String getPadding() {
        return padding;
    }

    /**
     * Returns the activation function of the layer.
     *
     * @return activation function of the layer. (IActivation)
     */
    public IActivation getActivation() {
        return activation;
    }

    public String getKernelInitializer() {
        return kernelInitializer;
    }

    public int getPad() {
        return pad;
    }

    public Layer<ZeroPadding2D> getZeroPadding2D() {
        return zeroPadding2D;
    }

    @Override
    public String toString() {
        String sb = "Conv2D{" +
                "filters=" + filters +
                ", kernelSize=" + kernelSize +
                ", strides=" + strides +
                ", padding='" + padding + '\'' +
                ", activation=" + activation.getClass().getSimpleName() +
                ", kernelInitializer='" + kernelInitializer + '\'' +
                ", params=(" + "W=" + Arrays.toString(getWeights().reshape(-1).toDoubleVector()) + ", b=" + Arrays.toString(getBiases().reshape(-1).toDoubleVector()) +
                "), grads=(" + "dW=" + Arrays.toString(getGradWeights().reshape(-1).toDoubleVector()) + ", db=" + Arrays.toString(getGradBiases().reshape(-1).toDoubleVector()) +
                "))";
        return sb;
    }

    /**
     * Saves the Layer's additional parameters to a file.
     * Writes the layer to a file using the following format:
     * <ul>
     *     <li>Layer class name (String)</li>
     *     <li>Input of the layer (INDArray)</li>
     *     <li>Output of the layer (INDArray)</li>
     *     <li>filters (int)</li>
     *     <li>kernelSize (int)</li>
     *     <li>strides (INDArray)</li>
     *     <li>padding (String)</li>
     *     <li>activation (String)</li>
     *     <li>kernelInitializer (String)</li>
     *     <li>isInitialized (boolean)</li>
     * @param dos
     * @throws IOException
     */
    @Override
    public void saveAdditional(DataOutputStream dos) throws IOException {
        if (!this.isInitialized) {
            throw new IOException("Layer is not initialized");
        }
        dos.writeInt(this.filters);
        dos.writeInt(this.kernelSize);
        // Save the size of the strides list
        dos.writeInt(this.strides.size());
        // Save each element in the strides list
        for (Integer stride : this.strides) {
            dos.writeInt(stride);
        }
        dos.writeUTF(this.padding);
        dos.writeUTF(this.activationType);
        dos.writeUTF(this.kernelInitializer);
        dos.writeBoolean(this.isInitialized);
        super.saveAdditional(dos);
    }

    @Override
    public void save(Datastore datastore) {
        if (this.getZeroPadding2D() != null) {
            this.getZeroPadding2D().save(datastore);
        }
        super.save(datastore);
    }

    @Override
    public Conv2D loadAdditional(DataInputStream dis) throws IOException {
        this.filters = dis.readInt();
        this.kernelSize = dis.readInt();
        // Read the size of the strides list
        int stridesSize = dis.readInt();
        this.strides = new ArrayList<>(stridesSize);
        // Read each element in the strides list
        for (int i = 0; i < stridesSize; i++) {
            this.strides.add(dis.readInt());
        }
        this.padding = dis.readUTF();
        this.activationType = dis.readUTF();
        this.activation = Activation.create(this.activationType);
        this.kernelInitializer = dis.readUTF();
        this.isInitialized = dis.readBoolean();
        super.loadAdditional(dis);
        this.setup(this.input);
        return this;
    }
}
