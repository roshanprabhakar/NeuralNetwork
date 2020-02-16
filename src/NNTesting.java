import java.util.Arrays;

public class NNTesting implements NetworkConstants {

    public static void main(String[] args) {

        Vector actual = new Vector(new double[]{0, 2});
        Vector input = new Vector(new double[]{1, 2, 3, 4, 5, 6});

        NeuralNetwork network = new NeuralNetwork(input.length(), 2, 2); //num output ignored for now
        network.train(input, actual);

        System.exit(0);

    }
}
