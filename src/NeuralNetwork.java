import javax.xml.stream.XMLInputFactory;
import java.util.ArrayList;

public class NeuralNetwork implements NetworkConstants {

    private ArrayList<Layer> network;
    private int inputSize;

    //learningRate * mazMomentum = largest possible step size
    private double learningRate;
    private double maxMomentum;

    //lower bounds for training
    private double minMomentum;
    private double lossSeparator;

    public NeuralNetwork(int[] layers, int inputSize, double learningRate, double maxMomentum, double minMomentum, double lossSeparator) {
        network = new ArrayList<>();
        network.add(new Layer(layers[0], inputSize));
        for (int layer = 1; layer < layers.length; layer++) {
            network.add(new Layer(layers[layer], layers[layer - 1]));
        }
        this.inputSize = inputSize;
        this.learningRate = learningRate;
        this.maxMomentum = maxMomentum;
        this.minMomentum = minMomentum;
        this.lossSeparator = lossSeparator;
    }

    public NeuralNetwork(int inputSize, int neuronsPerHidden, int numHiddenLayers, double learningRate, double maxMomentum, double minMomentum, double lossSeparator) {
        network = new ArrayList<>();
        network.add(new Layer(neuronsPerHidden, inputSize));
        for (int i = 0; i < numHiddenLayers - 1; i++) {
            network.add(new Layer(neuronsPerHidden, neuronsPerHidden));
        }
        this.inputSize = inputSize;
        this.learningRate = learningRate;
        this.maxMomentum = maxMomentum;
        this.minMomentum = minMomentum;
        this.lossSeparator = lossSeparator;
    }

    private void reinitializeWeightsAndBiases(double scope) {
        for (int layer = 0; layer < network.size(); layer++) {
            for (int neuron = 0; neuron < network.get(layer).length(); neuron++) {
                for (int weight = 0; weight < network.get(layer).get(neuron).getWeights().length(); weight++) {
                    network.get(layer).get(neuron).getWeights().set(weight, Math.random() * scope * Math.pow(-1, (int)((Math.random())*10)));
                }
                network.get(layer).get(neuron).updateBias(Math.random() * scope * Math.pow(-1, (int)((Math.random())*10)));
            }
        }
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

    //training when recalculating loss is not feasible
    public void train(ArrayList<NetworkData> trainingData) {
        double momentum = 1;
        int epoch = 0;
        double lossf = 0;
        double lossi = 1;
        while (momentum > minMomentum && lossi - lossf > lossSeparator) {
            lossi = cumulativeLoss(trainingData, this);
            NetworkGradient cumulativeLossGradient = new NetworkGradient();
            for (NetworkData data : trainingData) {
                cumulativeLossGradient.add(getGradient(data.getInput(), data.getOutput()));
            }
            NetworkGradient updateGradient = getUpdateVector(cumulativeLossGradient, momentum);
            updateWeightsAndBiases(updateGradient);
            double gradientMagnitude = getGradientMagnitude(cumulativeLossGradient);
            momentum = 2 * maxMomentum * (Perceptron.sigmoid(gradientMagnitude) - 0.5);
            if (momentum < minMomentum && epoch < 10) {
                reinitializeWeightsAndBiases(10);
                epoch = 0;
            }
            lossf = cumulativeLoss(trainingData, this);
            epoch++;
//            System.out.println("loss: " + lossf);
//            System.out.println("|gradient|: " + gradientMagnitude);
//            System.out.println("momentum: " + momentum);
        }
    }

    public double getGradientMagnitude(NetworkGradient networkOutput) {
        double magnitude = 0;
        for (ArrayList<Vector> layer : networkOutput.getdLossdWeights()) {
            for (Vector neuron : layer) {
                for (double weight : neuron.getVector()) {
                    magnitude += weight * weight;
                }
            }
        }
        for (ArrayList<Double> layer : networkOutput.getdLossdBiases()) {
            for (Double dlossdbias : layer) {
                magnitude += dlossdbias * dlossdbias;
            }
        }
        return Math.sqrt(magnitude);
    }

    public void updateWeightsAndBiases(NetworkGradient updateVector) {
        for (int layer = 0; layer < network.size(); layer++) {
            for (int neuron = 0; neuron < network.get(layer).length(); neuron++) {

                Perceptron perceptron = network.get(layer).get(neuron);

                for (int weight = 0; weight < perceptron.getWeights().length(); weight++) {
                    perceptron.getWeights().set(weight, perceptron.getWeights().get(weight) + updateVector.getdLossdWeights().get(layer).get(neuron).get(weight));
                }
                perceptron.updateBias(perceptron.getBias() + updateVector.getdLossdBiases().get(layer).get(neuron));
            }
        }
    }

    public NetworkGradient getUpdateVector(NetworkGradient gradient, double momentum) {

        ArrayList<ArrayList<Vector>> dLossdWeights = gradient.getdLossdWeights();
        ArrayList<ArrayList<Double>> dLossdBiases = gradient.getdLossdBiases();

        ArrayList<ArrayList<Vector>> weightUpdates = new ArrayList<>();
        ArrayList<ArrayList<Double>> biasUpdates = new ArrayList<>();

        for (int layer = 0; layer < network.size(); layer++) {
            ArrayList<Vector> thisLayerW = new ArrayList<>();
            ArrayList<Double> thisLayerB = new ArrayList<>();
            for (int neuron = 0; neuron < dLossdWeights.get(layer).size(); neuron++) {

                Vector updates = new Vector(dLossdWeights.get(layer).get(neuron).length());
                for (int weight = 0; weight < updates.length(); weight++) {
                    if (dLossdWeights.get(layer).get(neuron).get(weight) < 0) {
                        updates.set(weight, learningRate * momentum);
                    } else if (dLossdWeights.get(layer).get(neuron).get(weight) > 0) {
                        updates.set(weight, -learningRate * momentum);
                    } else {
                        updates.set(weight, Math.pow(-1, (int) (Math.random() * 10)) * learningRate * momentum);
                    }
                }
                thisLayerW.add(updates);

                if (dLossdBiases.get(layer).get(neuron) < 0) {
                    thisLayerB.add(learningRate * momentum);
                } else if (dLossdBiases.get(layer).get(neuron) > 0) {
                    thisLayerB.add(-learningRate * momentum);
                } else {
                    thisLayerB.add(Math.pow(-1, (int) (Math.random() * 10)) * learningRate * momentum);
                }
            }
            weightUpdates.add(thisLayerW);
            biasUpdates.add(thisLayerB);
        }
        return new NetworkGradient(weightUpdates, biasUpdates);
    }

    public NetworkGradient getGradient(Vector input, Vector correct) {

        //all information needed to calculate necessary partial derivatives
        ForwardPropOutput output = forwardProp(input);

        Vector prediction = output.getResultant();
        Vector[] neuronActivations = output.getIntermediaryMatrix();

        //derivatives of the loss with respect to every activation of the last layer
        Vector dLossdLastLayer = getLossDWRTLastLayer(correct, neuronActivations[neuronActivations.length - 1]);

        //derivatives of each each neuron in layer n to each neuron in layer n-1
        ArrayList<ArrayList<Vector>> dLayersdPreviousLayers = getLayerDerivatives(neuronActivations);

        //derivatives of each activation with respect to the weights of that neuron
        ArrayList<ArrayList<Vector>> dActivationsdWeights = getWeightDerivatives(neuronActivations, input);

        //derivatives of each activation with respect to the weights of that neuron
        ArrayList<ArrayList<Double>> dActivationsdBias = getBiasDerivatives(neuronActivations, input);

        //derivatives of the loss value with respect to each activation
        Vector[] dLossdActivations = getLossDWRTactivations(dLossdLastLayer, dLayersdPreviousLayers);

        //derivatives of the loss with respect to each weight - not verified yet
        ArrayList<ArrayList<Vector>> dLossdWeights = getWeightDWRTLoss(dLossdActivations, dActivationsdWeights);

        //derivatives of the loss with respect to each bias - not verified yet
        ArrayList<ArrayList<Double>> dLossdBiases = getBiasDWRTLoss(dLossdActivations, dActivationsdBias);

        return new NetworkGradient(dLossdWeights, dLossdBiases);
    }

    //Finds the derivative of each neuron's activation with respect to every weight within the neuron.
    //Written for implementation in some form of the chain rule.
    public ArrayList<ArrayList<Vector>> getWeightDerivatives(Vector[] neuronActivations, Vector input) {
        ArrayList<ArrayList<Vector>> weightDerivatives = new ArrayList<>();
        for (int layer = network.size() - 1; layer >= 0; layer--) {
            ArrayList<Vector> thisLayer = new ArrayList<>();
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

                    pDerivatives = new Vector(network.get(layer - 1).length());
                    guess = network.get(layer).get(neuron).unactivatedGuess(neuronActivations[layer - 1]);

                    for (int weight = 0; weight < network.get(layer - 1).length(); weight++) {
                        double factor = neuronActivations[layer - 1].get(weight);
                        pDerivatives.set(weight, Perceptron.sigmoidDerivative(guess) * factor);
                    }
                }
                thisLayer.add(pDerivatives);
            }
            weightDerivatives.add(0, thisLayer);
        }
        return weightDerivatives;
    }

    //Returns a vector matrix, where each matrix position corresponds to a neuron location
    //Vector indices in layer l line in index with the neurons of layer n - 1, in terms of of their derivatives
    //Specifically, at [2][0][3] is stored the derivative of the activation of neuron [2][0] with respect to the activation [1][3]
    //The first stored row is marked null to indicate that derivatives with respect to input values are obsolete
    public ArrayList<ArrayList<Vector>> getLayerDerivatives(Vector[] neuronActivations) {
        ArrayList<ArrayList<Vector>> layerDerivatives = new ArrayList<>();
        for (int layer = network.size() - 1; layer >= 1; layer--) {
            ArrayList<Vector> thisLayer = new ArrayList<>();
            for (int neuron = 0; neuron < network.get(layer).length(); neuron++) {
                Vector pDerivatives = new Vector(network.get(layer - 1).length());
                double guess = network.get(layer).get(neuron).unactivatedGuess(neuronActivations[layer - 1]);
                for (int weight = 0; weight < network.get(layer - 1).length(); weight++) {
                    double factor = network.get(layer).get(neuron).getWeights().get(weight);
                    pDerivatives.set(weight, Perceptron.sigmoidDerivative(guess) * factor);
                }
                thisLayer.add(pDerivatives);
            }
            layerDerivatives.add(0, thisLayer);
        }
        ArrayList<Vector> initialLayer = new ArrayList<>();
        for (int i = 0; i < network.get(0).length(); i++) {
            initialLayer.add(null);
        }
        layerDerivatives.add(0, initialLayer);
        return layerDerivatives;
    }

    //returns the derivative of each activation with respect to the bias associated with that same activation (matched by index)
    public ArrayList<ArrayList<Double>> getBiasDerivatives(Vector[] neuronActivations, Vector input) {
        ArrayList<ArrayList<Double>> biasDerivatives = new ArrayList<>();
        for (int layer = network.size() - 1; layer >= 0; layer--) {
            ArrayList<Double> thisLayer = new ArrayList<>();
            for (int neuron = 0; neuron < network.get(layer).length(); neuron++) {
                double bderivative;
                if (layer == 0) {
                    bderivative = Perceptron.sigmoidDerivative(network.get(layer).get(neuron).unactivatedGuess(input));
                } else {
                    bderivative = Perceptron.sigmoidDerivative(network.get(layer).get(neuron).unactivatedGuess(neuronActivations[layer - 1]));
                }
                thisLayer.add(bderivative);
            }
            biasDerivatives.add(0, thisLayer);
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
    public Vector[] getLossDWRTactivations(Vector lastLayerDerivatives, ArrayList<ArrayList<Vector>> layerDerivatives) {
        Vector[] activationDerivatives = new Vector[network.size()];
        activationDerivatives[network.size() - 1] = lastLayerDerivatives.copy();
        for (int layer = network.size() - 1; layer >= 1; layer--) {
            Vector pDerivatives = new Vector(network.get(layer - 1).length());
            for (int neuron2 = 0; neuron2 < network.get(layer - 1).length(); neuron2++) {
                double completeDerivative = 0;
                for (int neuron1 = 0; neuron1 < network.get(layer).length(); neuron1++) {
                    completeDerivative += layerDerivatives.get(layer).get(neuron1).get(neuron2) * lastLayerDerivatives.get(neuron1);
                }
                pDerivatives.set(neuron2, completeDerivative);
            }
            lastLayerDerivatives = pDerivatives.copy();
            activationDerivatives[layer - 1] = pDerivatives;
        }
        return activationDerivatives;
    }

    public ArrayList<ArrayList<Vector>> getWeightDWRTLoss(Vector[] activationDWRTLoss, ArrayList<ArrayList<Vector>> weightDWRTactivations) {
        ArrayList<ArrayList<Vector>> weightDWRTLoss = new ArrayList<>();
        for (int layer = 0; layer < network.size(); layer++) {
            ArrayList<Vector> thisLayer = new ArrayList<>();
            for (int neuron = 0; neuron < network.get(layer).length(); neuron++) {
                Vector weightDerivatives;
                if (layer == 0) {
                    weightDerivatives = new Vector(inputSize);
                    for (int weight = 0; weight < inputSize; weight++) {
                        weightDerivatives.set(weight, activationDWRTLoss[layer].get(neuron) * weightDWRTactivations.get(layer).get(neuron).get(weight));
                    }
                } else {
                    weightDerivatives = new Vector(network.get(layer).get(neuron).getWeights().length());
                    for (int weight = 0; weight < weightDerivatives.length(); weight++) {
                        weightDerivatives.set(weight, activationDWRTLoss[layer].get(neuron) * weightDWRTactivations.get(layer).get(neuron).get(weight));
                    }
                }
                thisLayer.add(weightDerivatives);
            }
            weightDWRTLoss.add(thisLayer);
        }
        return weightDWRTLoss;
    }

    public ArrayList<ArrayList<Double>> getBiasDWRTLoss(Vector[] activationDWRTLoss, ArrayList<ArrayList<Double>> biasDWRTactivations) {
        ArrayList<ArrayList<Double>> biasDWRTLoss = new ArrayList<>();
        for (int layer = 0; layer < network.size(); layer++) {
            ArrayList<Double> thisLayer = new ArrayList<>();
            for (int neuron = 0; neuron < network.get(layer).length(); neuron++) {
                thisLayer.add(biasDWRTactivations.get(layer).get(neuron) * activationDWRTLoss[layer].get(neuron));
            }
            biasDWRTLoss.add(thisLayer);
        }
        return biasDWRTLoss;
    }

    public static double cumulativeLoss(ArrayList<NetworkData> testSet, NeuralNetwork neuralNetwork) {
        double out = 0;
        for (NetworkData data : testSet) {
            out += computeLoss(neuralNetwork.forwardProp(data.getInput()).getResultant(), data.getOutput());
        }
        return out;
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
        for (int layer = 0; layer < network.size(); layer++) {
            System.out.println("---------------------------");
            System.out.println("LAYER: " + layer);
            System.out.println("---------------------------");
            for (int neuron = 0; neuron < network.get(layer).length(); neuron++) {
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

    public int numLayers() {
        return network.size();
    }

    public static class NetworkGradient {

        private ArrayList<ArrayList<Vector>> dLossdWeights;
        private ArrayList<ArrayList<Double>> dLossdBiases;

        public NetworkGradient(ArrayList<ArrayList<Vector>> dLossdWeights, ArrayList<ArrayList<Double>> dLossdBiases) {
            this.dLossdBiases = dLossdBiases;
            this.dLossdWeights = dLossdWeights;
        }

        public NetworkGradient() {
        }

        public void add(NetworkGradient other) {
            if (dLossdWeights == null || dLossdBiases == null) {
                dLossdWeights = other.getdLossdWeights();
                dLossdBiases = other.getdLossdBiases();
            } else {
                for (int layer = 0; layer < dLossdWeights.size(); layer++) {
                    for (int neuron = 0; neuron < dLossdWeights.get(layer).size(); neuron++) {
                        this.dLossdBiases.get(layer).set(neuron, this.dLossdBiases.get(layer).get(neuron) + other.dLossdBiases.get(layer).get(neuron));
                        this.dLossdWeights.get(layer).get(neuron).add(other.dLossdWeights.get(layer).get(neuron));
                    }
                }
            }
        }

        public ArrayList<ArrayList<Vector>> getdLossdWeights() {
            return dLossdWeights;
        }

        public ArrayList<ArrayList<Double>> getdLossdBiases() {
            return dLossdBiases;
        }
    }
}
