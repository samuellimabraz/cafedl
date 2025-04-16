package br.cafedl.neuralnetwork.database;

import br.cafedl.neuralnetwork.core.layers.Layer;
import br.cafedl.neuralnetwork.core.models.NeuralNetwork;
import com.mongodb.client.MongoClients;
import dev.morphia.Datastore;
import dev.morphia.Morphia;
import dev.morphia.query.Query;
import dev.morphia.query.filters.Filter;
import dev.morphia.query.filters.Filters;

import java.util.List;

public class NeuralNetworkService {
    private final static String MONGODB_URI = System.getenv("MONGODB_URI");
    private final Datastore datastore;

    public NeuralNetworkService() {
        this.datastore = Morphia.createDatastore(MongoClients.create(MONGODB_URI));
    }

    public void saveModel(NeuralNetwork model) {
        List<Layer> layers = model.getLayers();
        for (Layer<?> layer : layers) {
            layer.save(datastore);
        }
        // Agora salva a neural network
        datastore.save(model);
    }

    public NeuralNetwork loadModel(String modelName) {
        Filter filter = Filters.eq("name", modelName);
        Query<NeuralNetwork> query = datastore.find(NeuralNetwork.class).filter(filter);
        return query.first();
    }

    public List<NeuralNetwork> getAllModels() {
        return datastore.find(NeuralNetwork.class).iterator().toList();
    }
}