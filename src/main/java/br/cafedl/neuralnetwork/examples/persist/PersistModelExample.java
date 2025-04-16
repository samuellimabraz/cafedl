package br.cafedl.neuralnetwork.examples.persist;

import br.cafedl.neuralnetwork.core.layers.Layer;
import br.cafedl.neuralnetwork.core.models.NeuralNetwork;
import br.cafedl.neuralnetwork.database.NeuralNetworkService;

public class PersistModelExample {
    public static void saveModel(String filePath) throws Exception {
        // Carrega o modelo do arquivo
        NeuralNetwork loadedModel = NeuralNetwork.loadModel(filePath);

        System.out.println("Name: " + loadedModel.getName());
        System.out.println("Layers: " + loadedModel.getLayers().size());

        // Salva o modelo no banco de dados
        NeuralNetworkService service = new NeuralNetworkService();

        service.saveModel(loadedModel);

        NeuralNetwork loadedModelFromDB = service.loadModel(loadedModel.getName());

        // Verifica se o modelo foi salvo corretamente
        System.out.println("Name: " + loadedModelFromDB.getName());
        System.out.println("Layers size: " + loadedModelFromDB.getLayers().size());

        System.out.println("Connections");
        for (Layer l : loadedModelFromDB.getLayers()) {
            if (l.nextLayer != null)
                System.out.println(l.getName() + " -> " + l.nextLayer.getName());
        }

    }

    public static void main(String[] args) throws Exception {
        String filePath = "src/main/resources/br/deeplearning4java/neuralnetwork/examples/data/regression/saddle_function_model.zip";
        saveModel(filePath);
    }
}
