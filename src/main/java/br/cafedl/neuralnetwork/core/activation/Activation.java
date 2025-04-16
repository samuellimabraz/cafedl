package br.cafedl.neuralnetwork.core.activation;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.function.Supplier;

/**
 * Activation function factory.
 */
public class Activation {

    private static final Map<ActivateEnum, Supplier<IActivation>> activationMap = new HashMap<>();

    static {
        activationMap.put(ActivateEnum.SIGMOID, Sigmoid::new);
        activationMap.put(ActivateEnum.TANH, TanH::new);
        activationMap.put(ActivateEnum.RELU, ReLU::new);
        activationMap.put(ActivateEnum.SOFTMAX, Softmax::new);
        activationMap.put(ActivateEnum.SILU, SiLU::new);
        activationMap.put(ActivateEnum.LEAKY_RELU, LeakyReLU::new);
        activationMap.put(ActivateEnum.LINEAR, Linear::new);
    }

    private static final Map<String, ActivateEnum> labelMap = new HashMap<>();

    static {
        for (ActivateEnum e : ActivateEnum.values()) {
            labelMap.put(e.name(), e);
        }
    }

    private Activation() {
        throw new IllegalStateException("Utility class");
    }

    public static ActivateEnum valueOfLabel(String label) {
        return labelMap.get(label.toUpperCase());
    }

    /**
     * Returns an IActivation object of the specified type.
     * @param type the type of the IActivation object to create as a Type enum
     * @return an IActivation object of the specified type
     * @throws IllegalArgumentException if the specified type is not valid
     */
    public static IActivation create(ActivateEnum type) {
        return Optional.ofNullable(activationMap.get(type).get())
                .orElseThrow(() -> new IllegalArgumentException(
                        "Invalid type: " + type + ". Valid types are: " + Arrays.toString(ActivateEnum.values())));
    }

    /**
     * Returns an IActivation object of the specified label type.
     * @param type the type of the IActivation object to create, as a string
     * @return an IActivation object of the specified type
     * @throws IllegalArgumentException if the specified type is not valid
     */
    public static IActivation create(String type) {
        return create(valueOfLabel(type.toUpperCase()));
    }

}
