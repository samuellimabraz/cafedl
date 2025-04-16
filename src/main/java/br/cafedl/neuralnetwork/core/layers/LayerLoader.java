package br.cafedl.neuralnetwork.core.layers;

import java.io.DataInputStream;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Supplier;

public class LayerLoader {
    private static final Map<String, Supplier<Layer<?>>> layerLoaders = new HashMap<>();

    static {
        layerLoaders.put("Dense", Dense::new);
        layerLoaders.put("Flatten", Flatten::new);
        layerLoaders.put("Dropout", Dropout::new);
        layerLoaders.put("Conv2D", Conv2D::new);
        layerLoaders.put("MaxPooling2D", MaxPooling2D::new);
        layerLoaders.put("ZeroPadding2D", ZeroPadding2D::new);
    }

    public static Layer<?> load(DataInputStream dis) throws Exception {
        String layerType = dis.readUTF();
        //System.out.println("layerType: " + layerType);
        Supplier<Layer<?>> loader = layerLoaders.get(layerType);
        if (loader == null) {
            throw new Exception("Invalid layer type: " + layerType);
        }
        return loader.get().load(dis);
    }
}

