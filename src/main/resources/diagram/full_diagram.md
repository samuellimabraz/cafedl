```mermaid
---
config:
  look: handDrawn
  theme: dark
---
classDiagram
    namespace Activations {
        class IActivation {
            <<interface>>
            +forward(INDArray input) INDArray
            +backward(INDArray input) INDArray
        }
        class Activation {
            <<static>>
            -Map~ActivateEnum, Supplier~IActivation~~ activationMap
            -Map~String, ActivateEnum~ labelMap
            -Activation() 
            +static valueOfLabel(String label) ActivateEnum
            +static create(ActivateEnum type) IActivation
            +static create(String type) IActivation
        }
        class ActivateEnum {
            <<enumeration>>
            SIGMOID
            TANH
            RELU
            SOFTMAX
            SILU
            LEAKY_RELU
            LINEAR
        }
        class Sigmoid {
            +forward(INDArray input) INDArray
            +backward(INDArray input) INDArray
        }
        class TanH {
            +forward(INDArray input) INDArray
            +backward(INDArray input) INDArray
        }
        class ReLU {
            +forward(INDArray input) INDArray
            +backward(INDArray input) INDArray
        }
        class LeakyReLU {
            -double alpha
            +forward(INDArray input) INDArray
            +backward(INDArray input) INDArray
            +setAlpha(double alpha) void
        }
        class Linear {
            +forward(INDArray input) INDArray
            +backward(INDArray input) INDArray
        }
        class SiLU {
            -Sigmoid sigmoid
            +forward(INDArray input) INDArray
            +backward(INDArray input) INDArray
        }
        class Softmax {
            +forward(INDArray input) INDArray
            +backward(INDArray input) INDArray
        }
    }

    IActivation <|.. Sigmoid
    IActivation <|.. TanH
    IActivation <|.. ReLU
    IActivation <|.. LeakyReLU
    IActivation <|.. Linear
    IActivation <|.. SiLU
    IActivation <|.. Softmax
    Activation o--> ActivateEnum
    Activation <|.. IActivation
    SiLU ..> Sigmoid

    Dense *--> IActivation
    Conv2D *--> IActivation
    
    namespace models {
        class ModelBuilder {
            +layers: List<Layer>
            +add(layer:Layer): ModelBuilder
            +build(): NeuralNetwork
        }
        class NeuralNetwork {
            +id: ObjectId
            +name: String = "neural_network_" + UUID.randomUUID()
            #layers: List<Layer>
            -trainableLayers: List<TrainableLayer>
            -output: INDArray
            +NeuralNetwork(modelBuilder:ModelBuilder)
            #NeuralNetwork()
            +initTrainableLayers()
            +getId(): ObjectId
            +setName(name:String): void
            +getName(): String
            +predict(x:INDArray): INDArray
            +backPropagation(gradout:INDArray): void
            +getLayers(): List<Layer>
            +getTrainableLayers(): List<TrainableLayer>
            -setLayers(layers:List<Layer>): void
            +setTrainable(trainable:boolean): void
            +setInference(inference:boolean): void
            +saveModel(String filePath): void
            +static loadModel(String filePath): NeuralNetwork
        }
    }
    
    %% Relationships for models namespace
    ModelBuilder --> NeuralNetwork: builds >
    ModelBuilder *--> "1..*" Layer: contains >
    NeuralNetwork *--> "1..*" Layer: contains >
    NeuralNetwork *--> "0..*" TrainableLayer: contains >
    Optimizer o--> NeuralNetwork: manages >
    NeuralNetwork ..> LayerLoader: uses >
    
    namespace Layers {
        class Layer {
            <<abstract>>
            #INDArray input
            #INDArray output
            #boolean inference
            +INDArray forward(INDArray inputs)* 
            +INDArray backward(INDArray gradout)* 
            +Layer load(DataInputStream dis)
            +void save(DataOutputStream dos)
            +void saveAdditional(DataOutputStream dos)*
            +Layer loadAdditional(DataInputStream dis)*
            +void setInput(INDArray input)
            +INDArray getInput()
            +void setOutput(INDArray output)
            +INDArray getOutput()
            +void setInference(boolean inference)
            +void save(Datastore datastore)
        }
        class Flatten {
            +Flatten()
            +INDArray forward(INDArray inputs)
            +INDArray backward(INDArray gradout)
            +String toString()
            +Flatten loadAdditional(DataInputStream dis)
            +void saveAdditional(DataOutputStream dos)
        }
        class Dropout {
            -double dropoutRate
            -INDArray mask
            +Dropout(double dropoutRate)
            +Dropout()
            +INDArray forward(INDArray inputs)
            +INDArray backward(INDArray gradout)
            +String toString()
            +double getDropoutRate()
            +void setDropoutRate(double dropoutRate)
            +INDArray getMask()
            +void setMask(INDArray mask)
            +void saveAdditional(DataOutputStream dos)
            +Dropout loadAdditional(DataInputStream dis)
        }
        class ZeroPadding2D {
            #int padding
            +ZeroPadding2D(int padding)
            +ZeroPadding2D()
            +INDArray forward(INDArray inputs)
            +INDArray backward(INDArray gradout)
            +ZeroPadding2D loadAdditional(DataInputStream dis) 
            +void saveAdditional(DataOutputStream dos)
            +String toString()
        }
        class MaxPooling2D {
            -int poolSize
            -int stride
            +MaxPooling2D(int poolSize, int stride)
            +MaxPooling2D()
            +INDArray forward(INDArray inputs)
            +INDArray backward(INDArray gradout)
            +String toString()
            +MaxPooling2D loadAdditional(DataInputStream dis)
            +void saveAdditional(DataOutputStream dos)
        }
        class TrainableLayer {
            <<abstract>>
            #INDArray params
            #INDArray grads
            #boolean trainable
            #byte[] paramsData
            #byte[] gradsData
            +void setup(INDArray input)
            +INDArray getParams()
            +void setParams(INDArray params)
            +INDArray getGrads()
            +void setGrads(INDArray grads)
            +void setTrainable(boolean trainable)
            +boolean isTrainable()
            +void saveAdditional(DataOutputStream dos)
            +TrainableLayer loadAdditional(DataInputStream dis)
        }
        class Dense {
            -IActivation activation
            -String activationType
            -int units
            -boolean isInitialized
            -String kernelInitializer
            -double lambda
            +Dense(int units, IActivation activation, String kernelInitializer, double lambda)
            +Dense(int units, IActivation activation, String kernelInitializer)
            +Dense(int units, IActivation activation)
            +Dense(int units)
            +Dense()
            +INDArray getWeights()
            +INDArray getGradientWeights()
            +INDArray getBias()
            +INDArray getGradientBias()
            +void setup(INDArray inputs)
            +INDArray forward(INDArray inputs)
            +INDArray backward(INDArray gradout)
            +IActivation getActivation()
            +int getUnits()
            +String getKernelInitializer()
        }
        class Conv2D {
            #int filters
            #int kernelSize
            #List~Integer~ strides
            #String padding
            #IActivation activation
            #String activationType
            #String kernelInitializer
            #int pad
            #Layer~ZeroPadding2D~ zeroPadding2D
            #int m, nHInput, nWInput, nCInput
            #int nHOutput, nWOutput, nCOutput
            #boolean isInitialized
            #INDArray paddedInputs
            #INDArray weightsC, biasesC, aPrev, aSlicePrev
            #int[] vert_starts, horiz_starts
            +Conv2D(int filters, int kernelSize, List~Integer~ strides, String padding, IActivation activation, String kernelInitializer)
            +Conv2D(int filters, int kernelSize, String padding, IActivation activation, String kernelInitializer)
            +Conv2D(int filters, int kernelSize, IActivation activation)
            +Conv2D(int filters, int kernelSize)
            +Conv2D()
            +void setup(INDArray inputs)
            +INDArray forward(INDArray inputs)
            +INDArray backward(INDArray gradout)
            +INDArray getWeights()
            +INDArray getBiases()
            +void setWeights(INDArray weights)
            +void setBiases(INDArray biases)
            +INDArray getGradWeights()
            +INDArray getGradBiases()
        }
        class LayerLoader {
            -Map~String, Supplier~Layer~~ layerLoaders
            +static Layer load(DataInputStream dis)
        }
    }
    Layer <|-- TrainableLayer : >bind< T<-TrainableLayer
    TrainableLayer <|-- Dense : extends
    TrainableLayer <|-- Conv2D : extends
    Layer <|-- Flatten : >bind< T<-Flatten
    Layer <|-- Dropout : >bind< T<-Dropout
    Layer <|-- MaxPooling2D : >bind< T<-MaxPooling2D
    Layer <|-- ZeroPadding2D : >bind< T<-ZeroPadding2D
    Conv2D *-- ZeroPadding2D : uses
    LayerLoader --> Layer : creates

    namespace Optimizers {
        class Optimizer {
            <<abstract>>
            #NeuralNetwork neuralNetwork
            #LearningRateDecayStrategy learningRateDecayStrategy
            #double learningRate
            #Map~TrainableLayer, List~INDArray~~ auxParams
            #List~TrainableLayer~ trainableLayers
            -boolean initialized
            
            +Optimizer()
            #Optimizer(double learningRate)
            +Optimizer(LearningRateDecayStrategy learningRateDecayStrategy)
            +Optimizer(NeuralNetwork neuralNetwork)
            +setNeuralNetwork(NeuralNetwork neuralNetwork)
            #init()
            +update()
            +updateEpoch()
            #abstract List~INDArray~ createAuxParams(INDArray params)
            #abstract void updateRule(INDArray params, INDArray grads, List~INDArray~ auxParams)
        }
        
        %% Learning Rate Decay Strategies
        class LearningRateDecayStrategy {
            <<abstract>>
            #double decayPerEpoch
            #double learningRate
            
            +LearningRateDecayStrategy(double initialRate, double finalRate, int epochs)
            #abstract double calculateDecayPerEpoch(double initialRate, double finalRate, int epochs)
            +abstract double updateLearningRate()
        }
        

        class ExponentialDecayStrategy {
            +ExponentialDecayStrategy(double initialRate, double finalRate, int epochs)
            #double calculateDecayPerEpoch(double initialRate, double finalRate, int epochs)
            +double updateLearningRate()
        }

        class LinearDecayStrategy {
            +LinearDecayStrategy(double initialRate, double finalRate, int epochs)
            #double calculateDecayPerEpoch(double initialRate, double finalRate, int epochs)
            +double updateLearningRate()
        }
        
        %% SGD Optimizers
        class SGD {
            +SGD(double learningRate)
            +SGD(LearningRateDecayStrategy learningRateDecayStrategy)
            +SGD()
            #List~INDArray~ createAuxParams(INDArray params)
            #void updateRule(INDArray params, INDArray grads, List~INDArray~ auxParams)
        }
        
        class SGDMomentum {
            -double momentum
            -INDArray velocities
            
            +SGDMomentum(double learningRate, double momentum)
            +SGDMomentum(double learningRate)
            +SGDMomentum(LearningRateDecayStrategy learningRateDecayStrategy, double momentum)
            +SGDMomentum(LearningRateDecayStrategy learningRateDecayStrategy)
            #List~INDArray~ createAuxParams(INDArray params)
            #void updateRule(INDArray params, INDArray grads, List~INDArray~ auxParams)
        }
        
        class SGDNesterov {
            -double momentum
            -INDArray velocities
            
            +SGDNesterov(double learningRate, double momentum)
            +SGDNesterov(double learningRate)
            +SGDNesterov(LearningRateDecayStrategy learningRateDecayStrategy, double momentum)
            +SGDNesterov(LearningRateDecayStrategy learningRateDecayStrategy)
            #List~INDArray~ createAuxParams(INDArray params)
            #void updateRule(INDArray params, INDArray grads, List~INDArray~ auxParams)
        }
        
        class RegularizedSGD {
            -double alpha
            
            +RegularizedSGD(double learningRate, double alpha)
            +RegularizedSGD()
            +RegularizedSGD(double learningRate)
            +RegularizedSGD(LearningRateDecayStrategy learningRateDecayStrategy)
            +RegularizedSGD(LearningRateDecayStrategy learningRateDecayStrategy, double alpha)
            #List~INDArray~ createAuxParams(INDArray params)
            #void updateRule(INDArray params, INDArray grads, List~INDArray~ auxParams)
        }
        
        %% Adaptive optimizers
        class AdaGrad {
            -double eps
            -INDArray sumSquares
            
            +AdaGrad(double lr)
            +AdaGrad()
            +AdaGrad(LearningRateDecayStrategy learningRateDecayStrategy)
            #List~INDArray~ createAuxParams(INDArray params)
            #void updateRule(INDArray params, INDArray grads, List~INDArray~ auxParams)
        }
        
        class RMSProp {
            -double decayRate
            -double epsilon
            -INDArray accumulator
            
            +RMSProp(double learningRate, double decayRate, double epsilon)
            +RMSProp(LearningRateDecayStrategy learningRateDecayStrategy, double decayRate, double epsilon)
            +RMSProp(LearningRateDecayStrategy learningRateDecayStrategy)
            +RMSProp()
            +RMSProp(double learningRate, double decayRate)
            +RMSProp(double learningRate)
            #List~INDArray~ createAuxParams(INDArray params)
            #void updateRule(INDArray params, INDArray grads, List~INDArray~ auxParams)
        }
        
        class Adam {
            -double beta1
            -double beta2
            -double epsilon
            -INDArray m
            -INDArray v
            -int t
            
            +Adam(double learningRate, double beta1, double beta2, double epsilon)
            +Adam(double learningRate)
            +Adam()
            +Adam(LearningRateDecayStrategy learningRateDecayStrategy, double beta1, double beta2, double epsilon)
            +Adam(LearningRateDecayStrategy learningRateDecayStrategy)
            #List~INDArray~ createAuxParams(INDArray params)
            #void updateRule(INDArray params, INDArray grads, List~INDArray~ auxParams)
        }
        
        class AdaDelta {
            -double decayRate
            -double epsilon
            -INDArray accumulator
            -INDArray delta
            
            +AdaDelta(double decayRate, double epsilon)
            +AdaDelta(double decayRate)
            +AdaDelta()
            #List~INDArray~ createAuxParams(INDArray params)
            #void updateRule(INDArray params, INDArray grads, List~INDArray~ auxParams)
        }

    }

    Optimizer <|-- SGD
    Optimizer <|-- SGDMomentum
    Optimizer <|-- SGDNesterov
    Optimizer <|-- RegularizedSGD
    Optimizer <|-- AdaGrad
    Optimizer <|-- RMSProp
    Optimizer <|-- Adam
    Optimizer <|-- AdaDelta
    
    LearningRateDecayStrategy <|-- LinearDecayStrategy
    LearningRateDecayStrategy <|-- ExponentialDecayStrategy
    
    Optimizer o-- LearningRateDecayStrategy

    TrainableLayer <--o Optimizer: trainableLayers *

    namespace Losses {
        class ILossFunction {
            <<interface>>
            +INDArray forward(INDArray predicted, INDArray real)
            +INDArray backward(INDArray predicted, INDArray real)
        }

        %% Concrete classes implementing ILossFunction
        class MeanSquaredError {
            +INDArray forward(INDArray predictions, INDArray labels)
            +INDArray backward(INDArray predictions, INDArray labels)
        }

        class BinaryCrossEntropy {
            +INDArray forward(INDArray predictions, INDArray labels)
            +INDArray backward(INDArray predictions, INDArray labels)
        }

        class CategoricalCrossEntropy {
            -double eps
            +INDArray forward(INDArray predicted, INDArray real)
            +INDArray backward(INDArray predicted, INDArray real)
        }

        class SoftmaxCrossEntropy {
            -double eps
            -boolean singleClass
            -INDArray softmaxPreds
            -Softmax softmax
            +SoftmaxCrossEntropy()
            +SoftmaxCrossEntropy(double eps)
            +INDArray forward(INDArray predicted, INDArray real)
            +INDArray backward(INDArray predicted, INDArray real)
        }
    }

    ILossFunction <|.. MeanSquaredError
    ILossFunction <|.. BinaryCrossEntropy
    ILossFunction <|.. CategoricalCrossEntropy
    ILossFunction <|.. SoftmaxCrossEntropy

    namespace Metrics {
        class IMetric {
            <<interface>>
            +double evaluate(INDArray yTrue, INDArray yPred)
        }
        
        class MSE {
            +double evaluate(INDArray yTrue, INDArray yPred)
        }
        
        class RMSE {
            +double evaluate(INDArray yTrue, INDArray yPred)
        }
        
        class MAE {
            +double evaluate(INDArray yTrue, INDArray yPred)
        }
        
        class R2 {
            +double evaluate(INDArray yTrue, INDArray yPred)
        }
        
        class Accuracy {
            +double evaluate(INDArray yTrue, INDArray yPred)
        }

        class Precision {
            +double evaluate(INDArray yTrue, INDArray yPred)
        }
        
        class Recall {
            +double evaluate(INDArray yTrue, INDArray yPred)
        }
        
        class F1Score {
            +double evaluate(INDArray yTrue, INDArray yPred)
        }

    }

    IMetric <|.. MSE
    IMetric <|.. RMSE
    IMetric <|.. MAE
    IMetric <|.. R2 
    IMetric <|.. Accuracy 
    IMetric <|.. Precision 
    IMetric <|.. Recall 
    IMetric <|.. F1Score

    F1Score ..> Precision 
    F1Score ..> Recall 
    
    namespace Train {
        class TrainerBuilder {
            +batch: INDArray[2]
            +trainInputs: INDArray
            +trainTargets: INDArray
            +testInputs: INDArray
            +testTargets: INDArray
            +epochs: int = 100
            +batchSize: int = 32
            +earlyStopping: boolean = false
            +verbose: boolean = true
            +patience: int = 20
            +evalEvery: int = 10
            +trainRatio: double = 0.8
            
            +TrainerBuilder(model:NeuralNetwork, trainInputs:INDArray, trainTargets:INDArray, lossFunction:ILossFunction)
            +TrainerBuilder(model:NeuralNetwork, trainInputs:INDArray, trainTargets:INDArray, testInputs:INDArray, testTargets:INDArray, lossFunction:ILossFunction)
            +setOptimizer(optimizer:Optimizer): TrainerBuilder
            +setEpochs(epochs:int): TrainerBuilder
            +setBatchSize(batchSize:int): TrainerBuilder
            +setEarlyStopping(earlyStopping:boolean): TrainerBuilder
            +setPatience(patience:int): TrainerBuilder
            +setTrainRatio(trainRatio:double): TrainerBuilder
            +setEvalEvery(evalEvery:int): TrainerBuilder
            +setVerbose(verbose:boolean): TrainerBuilder
            +setMetric(metric:IMetric): TrainerBuilder
            +build(): Trainer
        }
        
        class Trainer {
            -model: NeuralNetwork
            -optimizer: Optimizer
            -lossFunction: ILossFunction
            -metric: IMetric
            -trainInputs: INDArray
            -trainTargets: INDArray
            -testInputs: INDArray
            -testTargets: INDArray
            -batch: INDArray[2]
            -epochs: int
            -batchSize: int
            -currentIndex: int
            -patience: int
            -evalEvery: int
            -earlyStopping: boolean
            -verbose: boolean
            -bestLoss: double
            -wait: int
            -threshold: double
            -trainLoss: double
            -valLoss: double
            -trainMetricValue: double
            -valMetricValue: double
            
            +Trainer(TrainerBuilder)
            +fit(): void
            +evaluate(): void
            -earlyStopping(): boolean
            -hasNextBatch(): boolean
            -getNextBatch(): void
            +splitData(inputs:INDArray, targets:INDArray, trainRatio:double): void
            +printDataInfo(): void
            +getTrainInputs(): INDArray
            +getTrainTargets(): INDArray
            +getTestInputs(): INDArray
            +getTestTargets(): INDArray
        }
    }
    
    %% Relationships for Train namespace
    TrainerBuilder --> Trainer: builds >
    TrainerBuilder o--> NeuralNetwork: model
    TrainerBuilder o--> ILossFunction: lossFunction
    TrainerBuilder o--> IMetric: metric
    TrainerBuilder o--> Optimizer: optimizer
    
    Trainer *--> NeuralNetwork: model
    Trainer *--> Optimizer: optimizer
    Trainer *--> ILossFunction: lossFunction
    Trainer *--> IMetric: metric

    namespace Data {
        class DataLoader {
            -INDArray trainData
            -INDArray testData
            +DataLoader(String trainDataPath, String testDataPath)
            +DataLoader(String trainX, String trainY, String testX, String testY)
            +static INDArray loadCsv(String csvFile)
            +INDArray getAllTrainImages()
            +INDArray getAllTestImages()
            +INDArray getAllTrainLabels()
            +INDArray getAllTestLabels()
            +INDArray getTrainImage(int index)
            +INDArray getTestImage(int index)
            +int getTrainLabel(int index)
            +int getTestLabel(int index)
            +INDArray getTrainData()
            +INDArray getTestData()
        }

        class Util {
            +static INDArray normalize(INDArray array)
            +static INDArray unnormalize(INDArray array)
            +static INDArray clip(INDArray array, double min, double max)
            +static INDArray[][] trainTestSplit(INDArray x, INDArray y, double trainSize)
            +static void printProgressBar(int current, int total)
            +static INDArray oneHotEncode(INDArray labels, int numClasses)
            +static WritableImage arrayToImage(INDArray imageArray, int WIDTH, int HEIGHT)
            +static WritableImage byteArrayToImage(byte[] byteArray)
            +static INDArray imageToINDArray(WritableImage writableImage, int width, int height)
            +static INDArray bytesToINDArray(byte[] bytes, int width, int height)
            +static INDArray confusionMatrix(INDArray predictions, INDArray labels)
        }

        class PlotDataPredict {
            +void plot2d(INDArray x, INDArray y, INDArray predict, String title)
            +void plot3dGridandScatter(INDArray x, INDArray y, INDArray predict, String title)
        }

        class DataProcessor {
            <<abstract>>
            +abstract void fit(INDArray data)
            +abstract INDArray transform(INDArray data)
            +abstract INDArray inverseTransform(INDArray data)
            +INDArray fitTransform(INDArray data)
        }

        class DataPipeline {
            -List<DataProcessor> processors
            +DataPipeline(List<DataProcessor> processors)
            +DataPipeline()
            +void add(DataProcessor processor)
            +void fit(INDArray data)
            +INDArray transform(INDArray data)
            +INDArray fitTransform(INDArray data)
            +INDArray inverseTransform(INDArray data)
            +List<DataProcessor> getProcessors()
        }

        class StandardScaler {
            -double mean, std
            -static final double EPSILON
            +void fit(INDArray data)
            +INDArray transform(INDArray data)
            +INDArray inverseTransform(INDArray data)
            +double getMean()
            +double getStd()
        }

        class MinMaxScaler {
            -INDArray min, max
            -final double minRange
            -final double maxRange
            +MinMaxScaler(double minRange, double maxRange)
            +MinMaxScaler()
            +void fit(INDArray data)
            +INDArray transform(INDArray data)
            +INDArray inverseTransform(INDArray data)
        }
    }

    namespace Database {
        class NeuralNetworkService {
            -final static String MONGODB_URI
            -final Datastore datastore
            +NeuralNetworkService()
            +void saveModel(NeuralNetwork model)
            +NeuralNetwork loadModel(String modelName)
            +List<NeuralNetwork> getAllModels()
        }
    }

    %% Relationships within Data namespace
    DataProcessor <|-- StandardScaler
    DataProcessor <|-- MinMaxScaler
    DataProcessor <|-- DataPipeline
    DataPipeline *--> "0..*" DataProcessor: contains

    %% Relationships between Data and other namespaces
    Trainer ..> Util: uses
    Trainer ..> DataLoader: uses
    TrainerBuilder ..> DataLoader: uses
    
    NeuralNetwork ..> Util: uses
    TrainableLayer ..> Util: uses
    
    models.NeuralNetwork ..> Util: uses for data processing
    Trainer *--> DataLoader: loads data
    
    %% Relationships for Database namespace
    NeuralNetwork --o Database.NeuralNetworkService: manages
    Layer ..> Database.NeuralNetworkService: uses for persistence
    NeuralNetworkService ..> Datastore: uses