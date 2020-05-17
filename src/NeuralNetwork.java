import java.util.ArrayList;
import java.util.Arrays;

//TODO write output layer calculations with neuron counts different than those of hidden layers
//TODO write gradient checking:
//      - compute pass starting on any neuron, ending on any neuron
//      - add a really h, forward pass again
//      - use the change in output neuron to determine an approximation for the derivative

public class NeuralNetwork implements NetworkConstants {

    private ArrayList<Layer> network;
    private int neuronsPerLayer;
    private int inputSize; //unused but probably necessary
    private int layers;

    public NeuralNetwork(int inputSize, int neuronsPerHidden, int numHiddenLayers) {
        network = new ArrayList<>();
        network.add(new Layer(neuronsPerHidden, inputSize));
        for (int i = 0; i < numHiddenLayers - 1; i++) {
            network.add(new Layer(neuronsPerHidden, neuronsPerHidden));
        }

        this.inputSize = inputSize;
        this.neuronsPerLayer = neuronsPerHidden;
        this.layers = numHiddenLayers;
    }

    //TODO develop this
    public NeuralNetwork(ArrayList<Layer> network) {
        this.network = network;
        this.layers = network.size();
    }

    //Customized for sigmoid activations active in every neuron
    public ForwardPropOutput forwardProp(Vector input) {
        Vector[] matrix = new Vector[network.size()];
        Vector passed = network.get(0).activations(input);
        matrix[0] = passed.copy();
        for (int i = 1; i < network.size(); i++) {
            passed = network.get(i).activations(passed);
            matrix[i] = passed.copy();
        }

        return new ForwardPropOutput(passed, matrix);
    }

    public ForwardPropOutput forwardProp(Vector input, int startLayer) {
        Vector[] matrix = new Vector[network.size() - startLayer];
        Vector passed = network.get(startLayer).activations(input);
        matrix[0] = passed.copy();
        for (int i = 1; i < matrix.length; i++) {
            passed = network.get(i + startLayer).activations(passed);
            matrix[i] = passed.copy();
        }
        return new ForwardPropOutput(passed, matrix);
    }

    //An implementation of gradient checking
    public double getApproximateLossDerivative(int layer, int neuron, int weight, Vector networkInput, Vector actual, double h) {

        ForwardPropOutput output = forwardProp(networkInput);
        Vector[] activations = output.getIntermediaryMatrix();

        ForwardPropOutput relativeOutput = null;
        if (layer == 0) {
            relativeOutput = forwardProp(networkInput, layer);
        } else {
            relativeOutput = forwardProp(activations[layer - 1], layer);
        }

        double initialLoss = computeLoss(relativeOutput.getResultant(), actual);


        Perceptron init = network.get(layer).get(neuron);
        init.getWeights().set(weight, init.getWeights().get(weight) + h);

        ForwardPropOutput changedOutput = null;
        if (layer == 0) {
            changedOutput = forwardProp(networkInput, layer);
        } else {
            changedOutput = forwardProp(activations[layer - 1], layer);
        }

        double finalLoss = computeLoss(changedOutput.getResultant(), actual);

        return (finalLoss - initialLoss) / h;

    }


    public void train(ArrayList<NetworkData> data) {
        for (NetworkData networkData : data) {
            train(networkData.getInput(), networkData.getCorrect());
        }
    }

    public void train(Vector input, Vector correct) {//TODO what value should be used to update each weight? It should be related to the derivatives themselves

        //all information needed to calculate necessary partial derivatives
        ForwardPropOutput output = forwardProp(input);

        Vector prediction = output.getResultant();
        Vector[] neuronActivations = output.getIntermediaryMatrix();

        //derivatives of the loss with respect to every activation of the last layer
        Vector dLossdLastLayer = getLossDWRTLastLayer(correct, neuronActivations[neuronActivations.length - 1]);

        //derivatives of each each neuron in layer n to each neuron in layer n-1
        Vector[][] dLayersdPreviousLayers = getLayerDerivatives(neuronActivations);

        //derivatives of each activation with respect to the weights of that neuron
        Vector[][] dActivationsdWeights = getWeightDerivatives(neuronActivations, input);

        //derivatives of each activation with respect to the weights of that neuron
        Double[][] dActivationsdBias = getBiasDerivatives(neuronActivations, input);

        //derivatives of the loss value with respect to each activation
        Vector[] dLossdActivations = getLossDWRTactivations(dLossdLastLayer, dLayersdPreviousLayers);

        //derivatives of the loss with respect to each weight - not verified yet
        Vector[][] dLossdWeights = getWeightDWRTLoss(dLossdActivations, dActivationsdWeights);

        //derivatives of the loss with respect to each bias - not verified yet
        Double[][] dLossdBiases = getBiasDWRTLoss(dLossdActivations, dActivationsdBias);


//        for (int i = 0; i < dLossdWeights.length; i++) {
//            System.out.println(Arrays.toString(dLossdWeights[i]));
//        }

        //Update all weights and biases
//        System.out.print("Loss before weight updates: ");
//        double lossi = computeLoss(prediction, correct);
//        System.out.println(lossi);

        for (int layerIndex = 0; layerIndex < layers; layerIndex++) {
            for (int neuronIndex = 0; neuronIndex < network.get(layerIndex).length(); neuronIndex++) {

                Perceptron neuron = network.get(layerIndex).get(neuronIndex);

                for (int weightIndex = 0; weightIndex < network.get(layerIndex).get(neuronIndex).getWeights().length(); weightIndex++) {

                    double weightSlope = dLossdWeights[layerIndex][neuronIndex].get(weightIndex);

                    //update transformations to induce change through dx/dy, not dy/dx where y=loss, x=weights vector
                    if (weightSlope > 0) {
                        neuron.updateWeight(weightIndex, neuron.getWeights().get(weightIndex) - (-0.01) * (1 / weightSlope));
                    } else if (weightSlope < 0) {
                        neuron.updateWeight(weightIndex, neuron.getWeights().get(weightIndex) + (-0.01) * (1 / weightSlope));
                    }

//                    System.out.print("Derivative of loss wrt this one weight: ");
//                    System.out.println(weightSlope);
                }

                double biasSlope = dLossdBiases[layerIndex][neuronIndex];

                //update transformations to induce change through dx/dy, not dy/dx where y=loss, x=weights vector
                if (biasSlope > 0) {
//                    neuron.updateBias(neuron.getBias() - (-0.01) * (1/biasSlope));
                    neuron.updateBias(neuron.getBias() - 0.001);
                } else if (biasSlope < 0) {
//                    neuron.updateBias(neuron.getBias() + (-0.01) * (1 / biasSlope));
                    neuron.updateBias(neuron.getBias() + 0.001);
                }
            }
        }

//        ForwardPropOutput newOutput = forwardProp(input);
//
//        System.out.print("Loss after weight updates: ");
//        double lossf = computeLoss(newOutput.getResultant(), correct);
//        System.out.println(lossf);
//
//        System.out.print("difference: ");
//        System.out.println(lossf - lossi);
    }

    //Finds the derivative of each neuron's activation with respect to every weight within the neuron.
    //Written for implementation in some form of the chain rule.
    public Vector[][] getWeightDerivatives(Vector[] neuronActivations, Vector input) {
        Vector[][] weightDerivatives = new Vector[layers][neuronsPerLayer];
        for (int layer = network.size() - 1; layer >= 0; layer--) {
            for (int neuron = 0; neuron < network.get(layer).length(); neuron++) {

                Vector pDerivatives;
                double guess;
                if (layer == 0) {

                    pDerivatives = new Vector(input.length());
                    guess = network.get(layer).get(neuron).unactivatedGuess(input);

                    for (int weight = 0; weight < input.length(); weight++) {
                        double factor = input.get(weight);
                        pDerivatives.set(weight, Perceptron.sigmoidDerivative(guess) * factor);
                    }

                } else {

                    pDerivatives = new Vector(neuronsPerLayer);
                    guess = network.get(layer).get(neuron).unactivatedGuess(neuronActivations[layer - 1]);

                    for (int weight = 0; weight < network.get(layer - 1).length(); weight++) {
                        double factor = neuronActivations[layer - 1].get(weight);
                        pDerivatives.set(weight, Perceptron.sigmoidDerivative(guess) * factor);
                    }
                }
                weightDerivatives[layer][neuron] = pDerivatives;
            }
        }
        return weightDerivatives;
    }

    //Returns a vector matrix, where each matrix position corresponds to a neuron location
    //Vector indices in layer l line in index with the neurons of layer n - 1, in terms of of their derivatives
    //Specifically, at [2][0][3] is stored the derivative of the activation of neuron [2][0] with respect to the activation [1][3]
    //The first stored row is marked null to indicate that derivatives with respect to input values are obsolete
    public Vector[][] getLayerDerivatives(Vector[] neuronActivations) {
        Vector[][] layerDerivatives = new Vector[layers][neuronsPerLayer];
        for (int layer = network.size() - 1; layer >= 1; layer--) {
            for (int neuron = 0; neuron < network.get(layer).length(); neuron++) {
                Vector pDerivatives = new Vector(neuronsPerLayer);
                double guess = network.get(layer).get(neuron).unactivatedGuess(neuronActivations[layer - 1]);
                for (int weight = 0; weight < network.get(layer - 1).length(); weight++) {
                    double factor = network.get(layer).get(neuron).getWeights().get(weight);
                    pDerivatives.set(weight, Perceptron.sigmoidDerivative(guess) * factor);

                }
                layerDerivatives[layer][neuron] = pDerivatives;
            }
        }
        return layerDerivatives;
    }

    //returns the derivative of each activation with respect to the bias associated with that same activation (matched by index)
    public Double[][] getBiasDerivatives(Vector[] neuronActivations, Vector input) {
        Double[][] biasDerivatives = new Double[layers][neuronsPerLayer];
        for (int layer = network.size() - 1; layer >= 0; layer--) {
            double bderivative;
            if (layer == 0) {
                for (int neuron = 0; neuron < network.get(layer).length(); neuron++) {
                    bderivative = Perceptron.sigmoidDerivative(network.get(layer).get(neuron).unactivatedGuess(input));
                    biasDerivatives[layer][neuron] = bderivative;
                }
            } else {
                for (int neuron = 0; neuron < network.get(layer).length(); neuron++) {
                    bderivative = Perceptron.sigmoidDerivative(network.get(layer).get(neuron).unactivatedGuess(neuronActivations[layer - 1]));
                    biasDerivatives[layer][neuron] = bderivative;
                }
            }
        }
        return biasDerivatives;
    }

    public Vector getLossDWRTLastLayer(Vector actual, Vector prediction) { //derivatives calculated for MSE
        Vector out = new Vector(actual.length());
        for (int i = 0; i < out.length(); i++) {
            out.set(i, -1 * (actual.get(i) - prediction.get(i)));
        }
        return out;
    }

    //returns a matrix of doubles representing the derivative of the loss function with respect to each activation at the current network configuration
    public Vector[] getLossDWRTactivations(Vector lastLayerDerivatives, Vector[][] layerDerivatives) {
        Vector[] activationDerivatives = new Vector[layers];
        activationDerivatives[layers - 1] = lastLayerDerivatives.copy();
        for (int layer = layers - 1; layer >= 1; layer--) {
            Vector pDerivatives = new Vector(neuronsPerLayer);
            for (int neuron2 = 0; neuron2 < neuronsPerLayer; neuron2++) {
                double completeDerivative = 0;
                for (int neuron1 = 0; neuron1 < neuronsPerLayer; neuron1++) {
                    completeDerivative += layerDerivatives[layer][neuron1].get(neuron2) * lastLayerDerivatives.get(neuron1);
                }
                pDerivatives.set(neuron2, completeDerivative);
            }
            lastLayerDerivatives = pDerivatives.copy();
            activationDerivatives[layer - 1] = pDerivatives;
        }
        return activationDerivatives;
    }

    public Vector[][] getWeightDWRTLoss(Vector[] activationDWRTLoss, Vector[][] weightDWRTactivations) {
        Vector[][] weightDWRTLoss = new Vector[layers][neuronsPerLayer];
        for (int layer = 0; layer < layers; layer++) {
            for (int neuron = 0; neuron < neuronsPerLayer; neuron++) {
                Vector weightDerivatives;
                if (layer == 0) {
                    weightDerivatives = new Vector(inputSize);
                    for (int weight = 0; weight < inputSize; weight++) {
                        weightDerivatives.set(weight, activationDWRTLoss[layer].get(neuron) * weightDWRTactivations[layer][neuron].get(weight));
                    }
                } else {
                    weightDerivatives = new Vector(neuronsPerLayer);
                    for (int weight = 0; weight < neuronsPerLayer; weight++) {
                        weightDerivatives.set(weight, activationDWRTLoss[layer].get(neuron) * weightDWRTactivations[layer][neuron].get(weight));
                    }
                    weightDWRTLoss[layer][neuron] = weightDerivatives;
                }
                weightDWRTLoss[layer][neuron] = weightDerivatives;
            }
        }
        return weightDWRTLoss;
    }

    public Double[][] getBiasDWRTLoss(Vector[] activationDWRTLoss, Double[][] biasDWRTactivations) {
        Double[][] biasDWRTLoss = new Double[layers][neuronsPerLayer];
        for (int layer = 0; layer < layers; layer++) {
            for (int neuron = 0; neuron < neuronsPerLayer; neuron++) {
                biasDWRTLoss[layer][neuron] = biasDWRTactivations[layer][neuron] * activationDWRTLoss[layer].get(neuron);
            }
        }
        return biasDWRTLoss;
    }

    public static double computeLoss(Vector predicted, Vector actual) {
        double sum = 0;
        assert predicted.length() == actual.length();
        for (int i = 0; i < predicted.length(); i++) {
            sum += (predicted.get(i) - actual.get(i)) * (predicted.get(i) - actual.get(i));
        }
        return sum * 0.5;
    }

    public void displayWeightsAndBiases() {
        for (int i = 0; i < 5; i++) System.out.println();
        System.out.println("DISPLAYING WEIGHTS");
        for (int layer = 0; layer < layers; layer++) {
            System.out.println("---------------------------");
            System.out.println("LAYER: " + layer);
            System.out.println("---------------------------");
            for (int neuron = 0; neuron < neuronsPerLayer; neuron++) {
                System.out.println("NEURON: " + neuron);
                System.out.println("WEIGHTS: " + network.get(layer).get(neuron).getWeights());
                System.out.println("BIAS: " + network.get(layer).get(neuron).getBias());
            }
        }
    }

    public void display() {
        for (int i = 0; i < inputSize; i++) {
            System.out.print("x" + i + "    ");
        }
        System.out.println();
        for (int i = 0; i < network.size(); i++) {
            System.out.println(network.get(i));
        }
    }

    public Perceptron getPerceptron(int layer, int perceptron) {
        return network.get(layer).get(perceptron);
    }

    public Layer getLayer(int i) {
        return network.get(i);
    }
}
