import java.util.ArrayList;

public class Main implements NetworkConstants {

    /**
     * Implementation of the perceptron algorithm for making market decisions based on stock data, currently being developed with the Iris DataSet
     */
    public static void main(String[] args) throws InterruptedException {

        DataSet set = new DataSet("TestData.txt");
        ArrayList<Data> data = set.getData(set.getDataSize(), 0);
        Perceptron perceptron = new Perceptron(DATA_LENGTH, TARGET, LEARNING_RATE, EPOCHS, POWER, false);
        perceptron.visualizeLoss(data);

        Thread.sleep(1000000);

//        DataSet set = new DataSet("TestData.txt");
//        Perceptron perceptron = new Perceptron(DATA_LENGTH, TARGET, LEARNING_RATE, EPOCHS, POWER);
//        perceptron.train(set.getData(set.getDataSize(), 0));
//        System.out.println(perceptron.loss(set.getData(set.getDataSize(), 0)));
//        System.out.println(perceptron.correct(set.getData(set.getDataSize(), 0)));
//        perceptron.visualizeLoss(set.getData(set.getDataSize(), 0));

//        DataSet set = new DataSet("TestData.txt");
//        System.out.println(isLinearlySeparable(set.getData(set.getDataSize(), 0), TARGET));


//        DataSet set = new DataSet("TestData.txt");
//        ArrayList<Data> data = set.getData(DATA_LENGTH, 0);
//        Graph graph = new Graph(data, NUM_DATA_GROUPS);
//
//        Perceptron perceptron = new Perceptron(DATA_LENGTH, TARGET, LEARNING_RATE, EPOCHS, POWER);
//        perceptron.train(data);

//        System.out.println(perceptron.loss(data));
//
//        System.out.println(perceptron.equation());
//        Thread.sleep(100000);
//
//        System.out.println(perceptron.test(data));
//
//
////        perceptron.visualTrain(data, NUM_DATA_GROUPS);
//
////        Graph graph = new Graph(data, 3);
////        graph.drawLine(graph.generateLine(perceptron.getWeights(), perceptron.getBias()));
////        graph.displayChart();
//
//        Thread.sleep(100000);
//        System.exit(0);

//        DataSet set = new DataSet("TestData.txt");
//        Perceptron perceptron = new Perceptron(DATA_LENGTH, TARGET, LEARNING_RATE, EPOCHS, POWER);
//        perceptron.train(set.getData(2, 0));
//        System.out.println("--------------------");
//        System.out.println("DATA: ");
//        DataSet.printSet(set.getData(2, 0));
//        System.out.println("PERCEPTRON: ");
//        System.out.print("Success: ");
//        System.out.println(perceptron.test(set.getData(2, 0)));
//        System.out.println("equation: ");
//        System.out.println(perceptron.equation());
    }

    public static void findBestAttributes(String filepath) throws InterruptedException { //filepath is the data file !!!BROKEN WITH SIGMOID SUCCESS CALCULATOR!!!
        DataSet set = new DataSet(filepath);
        ArrayList<ArrayList<Data>> dataSets = set.getDataSets(); //all sets of length r within length n data where r < n

        double bestSuccess = 0.0;
        ArrayList<Integer> attributes = new ArrayList<>();
        ArrayList<Data> bestDataSet = new ArrayList<>();
        for (int i = 0; i < dataSets.size(); i++) {
            ArrayList<Data> dataSet = dataSets.get(i);
            System.out.println((int) ((((double) i + 1) / dataSets.size()) * 100) + "% completed");
            Perceptron perceptron = new Perceptron(dataSet.get(0).size(), TARGET, LEARNING_RATE, EPOCHS, POWER, false);
            perceptron.train(dataSet);
            double success = perceptron.correct(dataSet);
            if (success > bestSuccess) {
                attributes = set.getPossibleCombinations().get(i);
                bestDataSet = dataSet;
                bestSuccess = success;
            }
        }

        System.out.println("best attributes: ");
        System.out.println(attributes);

        Perceptron perceptron = new Perceptron(attributes.size(), TARGET, LEARNING_RATE, EPOCHS, POWER, false);
        perceptron.train(bestDataSet);

        System.out.println("success: ");
        System.out.println(perceptron.correct(bestDataSet));
    }

    public static boolean isLinearlySeparable(ArrayList<Data> dataSet, String target) throws InterruptedException {
        Perceptron perceptron = new Perceptron(2, target, 0.01, 30000, 7, false);
        return perceptron.convergeTrain(dataSet, SEPARABLE);
    }
}
