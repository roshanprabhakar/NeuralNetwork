import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

public class NNTesting implements NetworkConstants {

    public static void main(String[] args) {

        Vector actual = new Vector(new double[]{0.5, 1});
        Vector input = new Vector(new double[]{-1, -0.5});

        IrisDataHandler handler = new IrisDataHandler("TestData.txt");
        HashMap<String, ArrayList<IrisData>> pairings = handler.getData(0.6);
        ArrayList<IrisData> trainingData = pairings.get("training");

        NeuralNetwork irisNetwork = new NeuralNetwork(4, 3, 2);

//        for (int epoch = 0; epoch < 10; epoch++) {
//            for (IrisData trainData : pairings.get("training")) {
//                irisNetwork.train(trainData.getInput(), trainData.getOutput());
//            }
//        }

        System.out.println();
        IrisData thisData2 = trainingData.get(1);
        irisNetwork.train(thisData2.getInput(), thisData2.getOutput());
        System.out.println("--------------------");
        System.out.println("a: " + thisData2.getOutput());
        System.out.println("p: " + irisNetwork.forwardProp(thisData2.getInput()).getResultant());
        System.out.println("input: " + thisData2.getInput());
        System.out.println("loss: " + NeuralNetwork.computeLoss(irisNetwork.forwardProp(thisData2.getInput()).getResultant(), thisData2.getOutput()));
        System.out.println("--------------------");
//        irisNetwork.display();

//        try {Thread.sleep(10000);} catch (InterruptedException ignored) {}

        System.out.println();
        IrisData thisData1 = trainingData.get(0);
        irisNetwork.train(thisData1.getInput(), thisData1.getOutput());
        System.out.println("--------------------");
        System.out.println("a: " + thisData1.getOutput());
        System.out.println("p: " + irisNetwork.forwardProp(thisData1.getInput()).getResultant());
        System.out.println("input: " + thisData1.getInput());
        System.out.println("loss: " + NeuralNetwork.computeLoss(irisNetwork.forwardProp(thisData1.getInput()).getResultant(), thisData1.getOutput()));
        System.out.println("--------------------");
        irisNetwork.display();



        System.exit(0);


        //Direct testing
        //Iris input data is of length 4
        NeuralNetwork network = new NeuralNetwork(2, 2, 2); //num output ignored for now
        network.train(input, actual);

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
}
