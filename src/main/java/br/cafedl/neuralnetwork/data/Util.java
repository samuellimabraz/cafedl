package br.cafedl.neuralnetwork.data;

import javafx.embed.swing.SwingFXUtils;
import javafx.scene.image.PixelWriter;
import javafx.scene.image.WritableImage;
import javafx.scene.paint.Color;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;

public class Util {

    /**
     * Normalize the data (Dup the array)
     *
     * @param array
     * @return normalized array -> xi / sum(x)
     */
    public static INDArray normalize(INDArray array) {
        INDArray normalized = array.dup();
        normalized.divi(normalized.sumNumber());
        return normalized;
    }

    /**
     * Unnormalize the data (Dup the array)
     *
     * @param array
     * @return unnormalized array -> xi * sum(x)
     */
    public static INDArray unnormalize(INDArray array) {
        INDArray unnormalized = array.dup();
        unnormalized.muli(unnormalized.sumNumber());
        return unnormalized;
    }

    /**
     * Clip values between min and max (In-place)
     *
     * @param array
     * @param min
     * @param max
     * @return
     */
    public static INDArray clip(INDArray array, double min, double max) {
        Transforms.max(array, min, false);
        Transforms.min(array, max, false);
        return array;
    }

    /**
     * Split the data into train and test sets
     *
     * @param x
     * @param y
     * @param trainSize
     * @return [xTrain, yTrain], [xTest, yTest] (INDArray[][])
     */
    public static INDArray[][] trainTestSplit(INDArray x, INDArray y, double trainSize) {
        int numTrain = (int) (x.size(0) * trainSize);
        if (numTrain == x.size(0)) {
            return new INDArray[][] { { x, y }, { x, y } };
        }
        INDArray xTrain = x.get(NDArrayIndex.interval(0, numTrain));
        INDArray xTest = x.get(NDArrayIndex.interval(numTrain, x.size(0)));
        INDArray yTrain = y.get(NDArrayIndex.interval(0, numTrain));
        INDArray yTest = y.get(NDArrayIndex.interval(numTrain, y.size(0)));
        return new INDArray[][] { { xTrain, yTrain }, { xTest, yTest } };
    }

    /**
     * Print a progress bar
     *
     * @param current
     * @param total
     */
    public static void printProgressBar(int current, int total) {
        int percent = (int) ((current / (double) total) * 100);
        int progressBars = percent / 2;

        StringBuilder progressBar = new StringBuilder(50);
        progressBar.append("\r[");

        for (int i = 0; i < 50; i++) {
            if (i < progressBars) {
                progressBar.append("=");
            } else if (i == progressBars) {
                progressBar.append(">");
            } else {
                progressBar.append(" ");
            }
        }

        progressBar.append("] ");
        progressBar.append(percent);
        progressBar.append("%");

        System.out.print(progressBar);
    }

    /**
     * One-hot encode the labels
     *
     * @param labels
     * @param numClasses
     * @return labels one-hot encoded
     *         ex: [0, 1, 2] -> [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
     */
    public static INDArray oneHotEncode(INDArray labels, int numClasses) {
        INDArray oneHotEncoded = Nd4j.zeros(DataType.DOUBLE, labels.rows(), numClasses);

        for (int i = 0; i < labels.rows(); i++) {
            int classIdx = labels.getInt(i);
            oneHotEncoded.putScalar(new int[]{i, classIdx}, 1);
        }

        return oneHotEncoded;
    }

    /**
     * Convert an array to an image
     *
     * @param imageArray
     * @param WIDTH
     * @param HEIGHT
     * @return image
     */
    public static WritableImage arrayToImage(INDArray imageArray, int WIDTH, int HEIGHT) {
        WritableImage image = new WritableImage(WIDTH, HEIGHT);
        PixelWriter pixelWriter = image.getPixelWriter();
        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                double colorValue = imageArray.getDouble(y * WIDTH + x);
                Color color = Color.gray(colorValue / 255.0);
                pixelWriter.setColor(x, y, color);
            }
        }
        return image;
    }

    public static WritableImage byteArrayToImage(byte[] byteArray) throws IOException {
        ByteArrayInputStream byteArrayInputStream = new ByteArrayInputStream(byteArray);
        BufferedImage bufferedImage = ImageIO.read(byteArrayInputStream);
        return SwingFXUtils.toFXImage(bufferedImage, null);
    }

    public static INDArray imageToINDArray(WritableImage writableImage, int width, int height) {
        // Convert the JavaFX image to a BufferedImage
        BufferedImage bufferedImage = SwingFXUtils.fromFXImage(writableImage, null);

        // Convert the BufferedImage to an OpenCV Mat
        int[] pixels = bufferedImage.getRGB(0, 0, bufferedImage.getWidth(), bufferedImage.getHeight(), null, 0, bufferedImage.getWidth());
        Mat mat = new Mat(bufferedImage.getHeight(), bufferedImage.getWidth(), CvType.CV_8UC3);
        byte[] data = new byte[pixels.length * (int)(mat.elemSize())];
        for (int i = 0; i < pixels.length; i++) {
            data[i*3] = (byte) ((pixels[i] >> 16) & 0xFF); // Red
            data[i*3 + 1] = (byte) ((pixels[i] >> 8) & 0xFF); // Green
            data[i*3 + 2] = (byte) ((pixels[i]) & 0xFF); // Blue
            //data[i*4 + 3] = (byte) ((pixels[i] >> 24) & 0xFF); // Alpha
        }
        mat.put(0, 0, data);

        // Resize the image to 28x28
        Mat resizedMat = new Mat();
        Imgproc.resize(mat, resizedMat, new Size(width, height), 0, 0, Imgproc.INTER_AREA);

        // Convert the resized image to grayscale
        Mat grayMat = new Mat();
        Imgproc.cvtColor(resizedMat, grayMat, Imgproc.COLOR_RGB2GRAY);

        double min = Core.minMaxLoc(grayMat).minVal;
        double max = Core.minMaxLoc(grayMat).maxVal;
        grayMat.convertTo(grayMat, CvType.CV_8UC1, 255.0/(max-min), -min * 255.0/(max-min));

        // Convert the grayscale image to an INDArray
        int totalBytes = (int) (grayMat.total() * grayMat.elemSize());
        byte[] buffer = new byte[totalBytes];
        grayMat.get(0, 0, buffer);

        // Convert buffer bytes to unsigned integers
        double[] unsignedBuffer = new double[buffer.length];
        for (int i = 0; i < buffer.length; i++) {
            unsignedBuffer[i] = ((int) buffer[i]) & 0xff;
        }

        //System.out.println("unsignedBuffer min - max: " + Arrays.stream(unsignedBuffer).min().getAsDouble() + " - " + Arrays.stream(unsignedBuffer).max().getAsDouble());

        return Nd4j.createFromArray(unsignedBuffer).castTo(DataType.DOUBLE).reshape(1, (long) width * height);
    }

    public static INDArray bytesToINDArray(byte[] bytes, int width, int height) throws IOException {
        // Convert the byte array to a BufferedImage
        BufferedImage bufferedImage = ImageIO.read(new ByteArrayInputStream(bytes));

        // Convert the BufferedImage to an OpenCV Mat
        int[] pixels = bufferedImage.getRGB(0, 0, bufferedImage.getWidth(), bufferedImage.getHeight(), null, 0, bufferedImage.getWidth());
        Mat mat = new Mat(bufferedImage.getHeight(), bufferedImage.getWidth(), CvType.CV_8UC3);
        byte[] data = new byte[pixels.length * (int)(mat.elemSize())];
        for (int i = 0; i < pixels.length; i++) {
            data[i*3] = (byte) ((pixels[i] >> 16) & 0xFF); // Red
            data[i*3 + 1] = (byte) ((pixels[i] >> 8) & 0xFF); // Green
            data[i*3 + 2] = (byte) ((pixels[i]) & 0xFF); // Blue
        }
        mat.put(0, 0, data);

        // Resize the image to 28x28
        Mat resizedMat = new Mat();
        Imgproc.resize(mat, resizedMat, new Size(width, height), 0, 0, Imgproc.INTER_AREA);

        // Convert the resized image to grayscale
        Mat grayMat = new Mat();
        Imgproc.cvtColor(resizedMat, grayMat, Imgproc.COLOR_RGB2GRAY);

        double min = Core.minMaxLoc(grayMat).minVal;
        double max = Core.minMaxLoc(grayMat).maxVal;
        grayMat.convertTo(grayMat, CvType.CV_8UC1, 255.0/(max-min), -min * 255.0/(max-min));

        // Convert the grayscale image to an INDArray
        int totalBytes = (int) (grayMat.total() * grayMat.elemSize());
        byte[] buffer = new byte[totalBytes];
        grayMat.get(0, 0, buffer);

        // Convert buffer bytes to unsigned integers
        double[] unsignedBuffer = new double[buffer.length];
        for (int i = 0; i < buffer.length; i++) {
            unsignedBuffer[i] = ((int) buffer[i]) & 0xff;
        }

        return Nd4j.createFromArray(unsignedBuffer).castTo(DataType.DOUBLE).reshape(1, (long) width * height);
    }

    public static INDArray confusionMatrix(INDArray predictions, INDArray labels) {
        int numClasses = predictions.columns();
        INDArray confusionMatrix = Nd4j.zeros(numClasses, numClasses);

        for (int i = 0; i < predictions.rows(); i++) {
            int trueClass = labels.getRow(i).argMax().getInt(0);
            int predictedClass = predictions.getRow(i).argMax().getInt(0);
            confusionMatrix.putScalar(trueClass, predictedClass, confusionMatrix.getDouble(trueClass, predictedClass) + 1);
        }

        return confusionMatrix;
    }

}
