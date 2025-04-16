package br.cafedl.neuralnetwork.examples.classification.image.qdraw;

import java.io.InputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

public class DataDownloader {
    private static final List<String> CLASS_NAMES = Arrays.asList(
            "cloud", "tedyy-bear", "basketball", "umbrella", "t-shirt", "baseball%20bat", "vase", "clock", "ladder", "tree");
    private static final String BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/";

    public static void main(String[] args) {
        try {
            downloadData();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void downloadData() throws Exception {
        String root = "src/main/resources/br/edu/unifei/ecot12/deeplearning4java/neuralnetwork/examples/data/quickdraw/npy";
        Files.createDirectories(Paths.get(root));

        System.out.println("Downloading...");

        for (String className : CLASS_NAMES) {
            String path = BASE_URL + className + ".npy";
            System.out.println(path);

            try (InputStream in = new URL(path).openStream()) {
                Files.copy(in, Paths.get(root, className + ".npy"));
            }
        }

        System.out.println("Loading...");
    }
}