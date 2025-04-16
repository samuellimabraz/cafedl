package br.cafedl.neuralnetwork.core.model;

import br.cafedl.neuralnetwork.core.layers.*;
import br.cafedl.neuralnetwork.core.models.NeuralNetwork;
import br.cafedl.neuralnetwork.database.NeuralNetworkService;
import com.mongodb.client.MongoClients;
import dev.morphia.Datastore;
import dev.morphia.Morphia;
import dev.morphia.query.Query;
import dev.morphia.query.filters.Filter;
import dev.morphia.query.filters.Filters;
import org.junit.jupiter.api.Test;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class SaveLoadDataBaseTest {
    private final static String DATABASE = "deeplearning4java";
    private final static String MONGODB_URI = "mongodb+srv://samuellimabraz:hibana22@cluster0.bo7cqjk.mongodb.net/?appName=Cluster0";

    @Test
    public void testSaveModel() throws Exception {
        // Carrega o modelo do arquivo
        String filePath = "src/main/resources/br/cafedl/neuralnetwork/examples/data/quickdraw/model/quickdraw-cnn.zip";
        NeuralNetwork loadedModel = NeuralNetwork.loadModel(filePath);
        loadedModel.setName("quickdraw-cnn");

        System.out.println("Name: " + loadedModel.getName());
        System.out.println("Layers: " + loadedModel.getLayers().size());

        // Salva o modelo no banco de dados
        NeuralNetworkService service = new NeuralNetworkService();

        service.saveModel(loadedModel);

        NeuralNetwork loadedModelFromDB = service.loadModel("quickdraw-cnn");

        // Verifica se o modelo foi salvo corretamente
        System.out.println("Name: " + loadedModelFromDB.getName());
        System.out.println("Layers: " + loadedModelFromDB.getLayers().size());

        System.out.println("Connections");
        for (Layer l : loadedModelFromDB.getLayers()) {
            if (l.nextLayer != null)
                System.out.println(l.getName() + " -> " + l.nextLayer.getName());
        }

        // Verifica se o modelo foi carregado corretamente
        List<TrainableLayer> trainableLayers = loadedModel.getTrainableLayers();
        List<TrainableLayer> trainableLayersFromDB = loadedModelFromDB.getTrainableLayers();

        System.out.println("trainableLayers.size(): " + trainableLayers.size());
        System.out.println("trainableLayersFromDB.size(): " + trainableLayersFromDB.size());

        assertEquals(trainableLayers.size(), trainableLayersFromDB.size());
        for (int i = 0; i < trainableLayers.size(); i++) {
            TrainableLayer trainableLayer = trainableLayers.get(i);
            TrainableLayer trainableLayerFromDB = trainableLayersFromDB.get(i);
            System.out.println("trainableLayerFromDB:" + trainableLayerFromDB.getName());
            assertEquals(trainableLayer.getName(), trainableLayerFromDB.getName());
            assertEquals(trainableLayer.getParams(), trainableLayerFromDB.getParams());
            assertEquals(trainableLayer.getGrads(), trainableLayerFromDB.getGrads());
        }

    }

    @Test
    public void testLoadModel() {
        final Datastore datastore = Morphia.createDatastore(MongoClients.create(MONGODB_URI));
        Class<Dense> f = datastore.getMapper().getClassFromCollection("dense");
        System.out.println("f: " + f);

        Filter filter = Filters.eq("name", "quickdraw_model-cnn");
        Query<NeuralNetwork> query = datastore.find(NeuralNetwork.class).filter(filter);
        List<NeuralNetwork> loadedModelsromDB = query.iterator().toList();
        System.out.println("loadedModelsromDB.size(): " + loadedModelsromDB.size());
        NeuralNetwork loadedModelFromDB = loadedModelsromDB.get(0);

        // Verifica se o modelo foi salvo corretamente
        System.out.println("Name: " + loadedModelFromDB.getName());
        System.out.println("Layers: " + loadedModelFromDB.getLayers().size());

        System.out.println("Connections");
        for (Layer l : loadedModelFromDB.getLayers()) {
            if (l.nextLayer != null)
                System.out.println(l.getName() + " -> " + l.nextLayer.getName());
        }

        List<TrainableLayer> trainableLayersFromDB = loadedModelFromDB.getTrainableLayers();
        for (int i = 0; i < trainableLayersFromDB.size(); i++) {
            TrainableLayer trainableLayerFromDB = trainableLayersFromDB.get(i);
            System.out.println("trainableLayerFromDB.getName(): " + trainableLayerFromDB.getName());
            System.out.println("trainableLayerFromDB.getParams(): " + trainableLayerFromDB.getParams());
            System.out.println("trainableLayerFromDB.getGrads(): " + trainableLayerFromDB.getGrads());
        }
    }

}
