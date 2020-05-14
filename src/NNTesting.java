public class NNTesting implements NetworkConstants {

    public static void main(String[] args) {

//        Vector actual = new Vector(new double[]{14, 3, 70, 19, 110, 11});
//        Vector input = new Vector(new double[]{2, 3, 3, 4});

        IrisDataHandler handler = new IrisDataHandler("Iris-setosa", "TestData.txt");

        //TODO shuffle this
        Vector[] inputList = handler.getInputData();
        Vector[] outputList = handler.getOutputData();

        //Iris input data is of length 4
        NeuralNetwork network = new NeuralNetwork(4, 1, 1); //num output ignored for now
//        network.display();

        for (int i = 0; i < inputList.length; i++) {
            network.train(inputList[i], outputList[i]);
        }

        System.exit(0);

    }
}
