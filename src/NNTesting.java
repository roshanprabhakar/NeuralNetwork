import org.jfree.chart.renderer.xy.VectorRenderer;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.function.DoubleToIntFunction;

public class NNTesting implements NetworkConstants {


    private static void printMnistMatrix(final MnistMatrix matrix) {
        System.out.println("label: " + matrix.getLabel());
        for (int r = 0; r < matrix.getNumberOfRows(); r++) {
            for (int c = 0; c < matrix.getNumberOfColumns(); c++) {
                System.out.print(matrix.getValue(r, c) + " ");
            }
            System.out.println();
        }
    }

    private static ArrayList<NetworkData> parseMnistMatrices(MnistMatrix[] data, double percent) {
        ArrayList<MnistMatrix> matrices = new ArrayList<>(Arrays.asList(data));
        Collections.shuffle(matrices);
        ArrayList<NetworkData> out = new ArrayList<>();
        for (int i = 0; i < percent * data.length; i++) {
            MnistMatrix dataPiece = matrices.get(i);
            Vector input = new Vector(dataPiece.getSingularized());
            Vector output = new Vector(10);
            output.set(dataPiece.getLabel(), 1);
            out.add(new NetworkData(input, output));
        }
        return out;
    }

    public static void main(String[] args) throws IOException {

        NeuralNetwork xorNetwork = new NeuralNetwork(new int[]{2}, 2, 0.01, 1, 0.01, 0.00001);

        NetworkData data1 = new NetworkData(new Vector(new double[]{0,0}), new Vector(new double[]{1, 0}));
        NetworkData data2 = new NetworkData(new Vector(new double[]{1,1}), new Vector(new double[]{1, 0}));
        NetworkData data3 = new NetworkData(new Vector(new double[]{1,0}), new Vector(new double[]{0, 1}));
        NetworkData data4 = new NetworkData(new Vector(new double[]{0,1}), new Vector(new double[]{0, 1}));
        ArrayList<NetworkData> trainingData = new ArrayList<>() {
            {
                add(data1); add(data2); add(data3); add(data4);
            }
        };

        xorNetwork.train(trainingData);
        for (NetworkData data : trainingData) {
            System.out.println("--------------");
            ForwardPropOutput output = xorNetwork.forwardProp(data.getInput());
            System.out.println("a: " + data.getOutput());
            System.out.println("n: " + output.getResultant().getNetworkOutputVector());
            System.out.println("p: " + output.getResultant());
            System.out.println("--------------");
        }


        System.exit(0);


        System.out.println("parsing mnist set...");
        MnistMatrix[] mnistMatrix = new MnistDataReader().readData("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte");
        ArrayList<NetworkData> data = parseMnistMatrices(mnistMatrix, 0.01);
        System.out.println("...finished parsing mnist data");

        NeuralNetwork mnistNetwork = new NeuralNetwork(new int[]{28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 10}, 28 * 28, 0.01, 400, 0.001, 0.001);
        System.out.println("training...");
        mnistNetwork.train(data);
        System.out.println("...training complete");
//        mnistNetwork.display();

        //testing network on mnist data
        double correct = 0;
        for (NetworkData dataPiece : data) {
            System.out.println("--------------");
//            System.out.println("i: " + dataPiece.getInput());
            ForwardPropOutput output = mnistNetwork.forwardProp(dataPiece.getInput());
            System.out.println("a: " + dataPiece.getOutput());
            System.out.println("n: " + output.getResultant().getNetworkOutputVector());
            System.out.println("p: " + output.getResultant());
            System.out.println("l: " + NeuralNetwork.computeLoss(output.getResultant(), dataPiece.getOutput()));
            System.out.println("--------------");
            if (output.getResultant().getNetworkOutputVector().equals(dataPiece.getOutput())) {
                correct++;
            }
        }
        mnistNetwork.display();

        System.out.println("Summary");
        System.out.println("final loss: " + NeuralNetwork.cumulativeLoss(data, mnistNetwork));
        System.out.println("% correct: " + (correct / data.size()) + " : " + (int)correct + "/" + data.size());

        System.exit(0);

// -------------------------------------------------------------------------------------------------------------------
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
//
// -------------------------------------------------------------------------------------------------------------------
//
//        IrisDataHandler handler = new IrisDataHandler("TestData.csv");
//        HashMap<String, ArrayList<NetworkData>> pairings = handler.getData(0.9);
//
////        Vector trainingInput = new Vector(new double[]{7.7, 2.6, 6.9, 2.3});
////        Vector output = new Vector(new double[]{0, 1, 0});
//
//        NeuralNetwork irisNetwork = new NeuralNetwork(new int[]{3}, 4, 0.01, 2, 0.01, 0.0001);
//
////        for (int epoch = 0; epoch < 2; epoch++) {
////            Collections.shuffle(pairings.get("training"));
//        irisNetwork.train(pairings.get("training"));
////        }
//
//        irisNetwork.display();
//
//        System.out.println();
//        double correct = 0;
//        for (NetworkData testData : pairings.get("testing")) {
//            System.out.println("----------------------");
//            System.out.println("a: " + testData.getOutput());
//            ForwardPropOutput output = irisNetwork.forwardProp(testData.getInput());
//            System.out.println("n: " + output.getResultant().getNetworkOutputVector());
//            System.out.println("p: " + output.getResultant());
//            System.out.println("i: " + testData.getInput());
//            System.out.println("l: " + NeuralNetwork.computeLoss(output.getResultant(), testData.getOutput()));
//            System.out.println("----------------------");
//            if (output.getResultant().getNetworkOutputVector().equals(testData.getOutput())) {
//                correct++;
//            }
//        }
//        System.out.println("Summary: ");
//        System.out.println("% Correct: " + correct / pairings.get("testing").size());
//        System.out.println("Cumulative loss: " + NeuralNetwork.cumulativeLoss(pairings.get("testing"), irisNetwork));
//
//        System.exit(0);
// -------------------------------------------------------------------------------------------------------------------

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
