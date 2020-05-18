import java.util.Arrays;

public class NNTesting implements NetworkConstants {

    public static void main(String[] args) {

        Vector actual = new Vector(new double[]{0.5, 1});
        Vector input = new Vector(new double[]{-1, -0.5});

//        IrisDataHandler handler = new IrisDataHandler("TestData.txt");
//        ArrayList<IrisData> pairings = handler.getData();
        
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
