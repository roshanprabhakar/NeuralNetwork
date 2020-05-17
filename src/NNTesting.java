import java.util.ArrayList;
import java.util.Collections;

public class NNTesting implements NetworkConstants {

    public static void main(String[] args) {

//        Vector actual = new Vector(new double[]{5, 1});
//        Vector input = new Vector(new double[]{2});

        IrisDataHandler handler = new IrisDataHandler("TestData.txt");
        ArrayList<IrisData> pairings = handler.getData();
        
        //Iris input data is of length 4
        NeuralNetwork network = new NeuralNetwork(4, 1, 3); //num output ignored for now

        for (int i = 0; i < 100; i++) {
            Collections.shuffle(pairings);
            for (IrisData data : pairings) {
                network.train(data.getInput(), data.getOutput());
            }
        }

        for (IrisData data : pairings) {
            System.out.println();
            System.out.println("---------------");
            System.out.println("input: " + data.getInput());
            System.out.println("label: " + data.getOutput());
            System.out.println("predicted: " + network.forwardProp(data.getInput()).getResultant());
            System.out.println("---------------");
            System.out.println();
        }

//        System.out.println("The network: ");
//        network.display();

//        System.out.println();
//        System.out.println("Appropriate weight derivatives: ");
//        network.train(input, actual);
//
//        System.out.println();
//        System.out.println("Derivative of Loss with respect to specified weight through gradient approximation");
//        System.out.println(network.getApproximateLossDerivative(0, 0, 0, input, actual, 0.00001));




//        network.train(input, actual);


//        for (int epoch = 0; epoch < 1; epoch++) {
//            Collections.shuffle(pairings);
//            for (IrisData pairing : pairings) {
//                network.train(pairing.getInput(), pairing.getOutput());
//            }
//        }

//        System.out.println();
//        System.out.println(pairings.get(18).getOutput());
//        System.out.println(network.forwardProp(pairings.get(0).getInput()).getResultant());
//
        System.exit(0);

    }
}
