import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.function.DoubleToIntFunction;

public class NNTesting implements NetworkConstants {

    public static void main(String[] args) {

        Vector actual = new Vector(new double[]{0.5, 1});
        Vector input = new Vector(new double[]{-1, -0.5});

        IrisDataHandler handler = new IrisDataHandler("TestData.txt");
        HashMap<String, ArrayList<NetworkData>> pairings = handler.getData(0.6);

//        Vector trainingInput = new Vector(new double[]{7.7, 2.6, 6.9, 2.3});
//        Vector output = new Vector(new double[]{0, 1, 0});

        //TODO network works better with less training iterations? with too many, it converges on a single output for any given input - with cumulative gd per test point
        NeuralNetwork irisNetwork = new NeuralNetwork(4, 9, 2);

//        System.exit(0);
//        irisNetwork.train(trainingInput, output);

//        for (int epoch = 0; epoch < 10; epoch++) {
//        for (NetworkData trainData : pairings.get("training")) {
//            irisNetwork.train(trainData.getInput(), trainData.getOutput());
//            System.out.println("generalizability: " + evaluateNN(irisNetwork, pairings.get("testing")));
//        }

        for (int epoch = 0; epoch < 100; epoch++) {
            Collections.shuffle(pairings.get("training"));
            irisNetwork.train(pairings.get("training"));
        }
//        }

        irisNetwork.display();

        System.out.println();
        for (NetworkData testData : pairings.get("testing")) {
            System.out.println("----------------------");
            System.out.println("a: " + testData.getOutput());
            ForwardPropOutput output = irisNetwork.forwardProp(testData.getInput());
            System.out.println("p: " + output.getResultant());
            System.out.println("i: " + testData.getInput());
            System.out.println("l: " + NeuralNetwork.computeLoss(output.getResultant(), testData.getOutput()));
            System.out.println("----------------------");
        }

        System.exit(0);


        //Direct testing
        //Iris input data is of length 4
        NeuralNetwork network = new NeuralNetwork(2, 2, 2); //num output ignored for now
        network.train(input, actual, 1);

        System.exit(0);

        System.out.println("The network");
        network.display();

        NeuralNetwork.NetworkGradient gradient = network.getGradient(input, actual);
        NeuralNetwork.NetworkGradient updateVector = network.getUpdateVector(gradient, 0.001);

        Vector[][] dLossdWeights = gradient.getdLossdWeights();

        System.out.println();
        System.out.println("Calculated weight derivatives");
        for (Vector[] layer : dLossdWeights) {
            System.out.println(Arrays.toString(layer));
        }

        int layer = 1;
        int neuron = 0;
        int weight = 0;
        double step = 0.0000001;

        System.out.println();
        System.out.println("Derivative of Loss with respect to specified weight through gradient approximation");
        double gradientApprox = network.getApproximateLossDerivative(layer, neuron, weight, input, actual, step);
        System.out.println(gradientApprox);

        System.out.println();
        System.out.println("Difference");
        System.out.println(Math.abs(gradientApprox - dLossdWeights[layer][neuron].get(weight)));

        System.out.println();
        System.out.println("Weights update vector");
        for (int i = 0; i < updateVector.getdLossdWeights().length; i++) {
            System.out.println(Arrays.toString(updateVector.getdLossdWeights()[i]));
        }

        System.exit(0);

    }

    public static double evaluateNN(NeuralNetwork network, ArrayList<NetworkData> testDataList) {
        double sum = 0;
        for (NetworkData testData : testDataList) {
            Vector real = testData.getOutput();
            Vector predicted = network.forwardProp(testData.getInput()).getResultant();
            double loss = NeuralNetwork.computeLoss(predicted, real);
            sum += loss;
        }
        return sum;
    }
}
