import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.function.DoubleToIntFunction;


//TODO Question: should cumulative or differential gradient descent be used?
//TODO Potential Solution: implement a gradient calculation that takes the cumulative loss of the training set as the loss output, instead of individual losses for each training point
public class NNTesting implements NetworkConstants {

    public static void main(String[] args) {

////        Vector actual = new Vector(new double[]{0.5, 1});
////        Vector input = new Vector(new double[]{-1, -0.5});
//
//        //Network for the xor operation
//        NetworkData data1 = new NetworkData(new Vector(new double[]{0,0}), new Vector(new double[]{0}));
//        NetworkData data2 = new NetworkData(new Vector(new double[]{1,1}), new Vector(new double[]{0}));
//        NetworkData data3 = new NetworkData(new Vector(new double[]{1,0}), new Vector(new double[]{1}));
//        NetworkData data4 = new NetworkData(new Vector(new double[]{0,1}), new Vector(new double[]{1}));
//        ArrayList<NetworkData> trainingData = new ArrayList<>() {
//            {
//                add(data1); add(data2); add(data3); add(data4);
//            }
//        };
//
//        NeuralNetwork xorNetwork = new NeuralNetwork(2, 1, 1);
//        xorNetwork.train(trainingData);
//
//        for (NetworkData data : trainingData) {
//            System.out.println("--------------");
//            System.out.println("i: " + data.getInput());
//            ForwardPropOutput output = xorNetwork.forwardProp(data.getInput());
//            System.out.println("p: " + output.getResultant());
//            System.out.println("a: " + data.getOutput());
//            System.out.println("l: " + NeuralNetwork.computeLoss(output.getResultant(), data.getOutput()));
//            System.out.println("--------------");
//        }

        IrisDataHandler handler = new IrisDataHandler("TestData.csv");
        HashMap<String, ArrayList<NetworkData>> pairings = handler.getData(0.9);

//        Vector trainingInput = new Vector(new double[]{7.7, 2.6, 6.9, 2.3});
//        Vector output = new Vector(new double[]{0, 1, 0});

        NeuralNetwork irisNetwork2 = new NeuralNetwork(4, 3, 1);
        NeuralNetwork irisNetwork = new NeuralNetwork(new int[]{2,3}, 4);

//        for (int epoch = 0; epoch < 2; epoch++) {
//            Collections.shuffle(pairings.get("training"));
            irisNetwork.train(pairings.get("training"), 0.00001);
//        }

        irisNetwork.display();

        System.out.println();
        double correct = 0;
        for (NetworkData testData : pairings.get("testing")) {
            System.out.println("----------------------");
            System.out.println("a: " + testData.getOutput());
            ForwardPropOutput output = irisNetwork.forwardProp(testData.getInput());
            System.out.println("n: " + output.getResultant().getNetworkOutputVector());
            System.out.println("p: " + output.getResultant());
            System.out.println("i: " + testData.getInput());
            System.out.println("l: " + NeuralNetwork.computeLoss(output.getResultant(), testData.getOutput()));
            System.out.println("----------------------");
            if (output.getResultant().getNetworkOutputVector().equals(testData.getOutput())) {
                correct++;
            }
        }
        System.out.println("Summary: ");
        System.out.println("% Correct: " + correct / pairings.get("testing").size());
        System.out.println("Cumulative loss: " + NeuralNetwork.cumulativeLoss(pairings.get("testing"), irisNetwork));

        System.exit(0);


//        //Direct testing
//        //Iris input data is of length 4
//        NeuralNetwork network = new NeuralNetwork(2, 2, 2); //num output ignored for now
//        network.train(input, actual, 1);
//
//        System.exit(0);
//
//        System.out.println("The network");
//        network.display();
//
//        NeuralNetwork.NetworkGradient gradient = network.getGradient(input, actual);
//        NeuralNetwork.NetworkGradient updateVector = network.getUpdateVector(gradient, 0.001);
//
//        Vector[][] dLossdWeights = gradient.getdLossdWeights();
//
//        System.out.println();
//        System.out.println("Calculated weight derivatives");
//        for (Vector[] layer : dLossdWeights) {
//            System.out.println(Arrays.toString(layer));
//        }
//
//        int layer = 1;
//        int neuron = 0;
//        int weight = 0;
//        double step = 0.0000001;
//
//        System.out.println();
//        System.out.println("Derivative of Loss with respect to specified weight through gradient approximation");
//        double gradientApprox = network.getApproximateLossDerivative(layer, neuron, weight, input, actual, step);
//        System.out.println(gradientApprox);
//
//        System.out.println();
//        System.out.println("Difference");
//        System.out.println(Math.abs(gradientApprox - dLossdWeights[layer][neuron].get(weight)));
//
//        System.out.println();
//        System.out.println("Weights update vector");
//        for (int i = 0; i < updateVector.getdLossdWeights().length; i++) {
//            System.out.println(Arrays.toString(updateVector.getdLossdWeights()[i]));
//        }
//
//        System.exit(0);

    }
}
