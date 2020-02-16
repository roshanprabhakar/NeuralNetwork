import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import java.util.ArrayList;
import java.util.Collections;

public class Perceptron {

    private int dataLength;

    private Vector weights;
    private double bias;

    private String classifyFor;
    private double learningRate;
    private int power;
    private int epochs;

    private static final int TIME_OUT = 20;

    private static final String[] visualizableDimensions = {"x", "y", "z", "a"};

    public Perceptron(int dataLength, String classifyFor, double learningRate, int epochs, int power, boolean randomInitializations) {

        if (randomInitializations) {
            weights = new Vector(dataLength * power, true); //don't want to replace vector initializations everywhere, just created a second constructor
            bias = Math.random();
        } else {
            weights = new Vector(dataLength * power);
            bias = 0;
        }

        this.classifyFor = classifyFor;
        this.learningRate = learningRate;
        this.power = power;
        this.epochs = epochs;
        this.dataLength = dataLength;
    }

    public void visualizeLoss(ArrayList<Data> trainingData) throws InterruptedException {
        XYSeries loss = new XYSeries("loss");
        XYSeries correct = new XYSeries("% correct");
        XYSeriesCollection collection = new XYSeriesCollection();
        for (int rep = 0; rep < epochs; rep++) {
            loss.add(rep, loss(trainingData));
            correct.add(rep, correct(trainingData) * 10); //scaled for visuability
            for (Data data : trainingData) {
                train(data);
            }
        }
        collection.addSeries(loss);
        collection.addSeries(correct);

        Graph.displayChart(Graph.createChart(collection));
    }

    public double loss(ArrayList<Data> testData) {
        double loss = 0;
        for (Data data : testData) {
            loss += (map(data.getId()) - guess(data.getVector())) *
                    (map(data.getId()) - guess(data.getVector()));
//            System.out.println(loss);
        }
        return loss;
    }

    //remove automatic epochs, add manual epochs: gives a numeric success measurement
    //train all datesets to be tested with the same number epoch
    public void train(ArrayList<Data> trainingDataSet) throws InterruptedException {
        for (int rep = 0; rep < epochs; rep++) {
            for (Data trainingData : trainingDataSet) {
                train(trainingData);
            }
        }
    }

    public boolean convergeTrain(ArrayList<Data> trainingDataSet, double percentConvergence) throws InterruptedException { //returns true once dataSet is separated
        long start = System.currentTimeMillis();
        double correct = 0;
        while (correct / trainingDataSet.size() < percentConvergence && !timedOut(start, power * 10)) {
            correct = 0;
            for (Data trainingData : trainingDataSet) {
                train(trainingData);
            }
            for (Data trainingData : trainingDataSet) {
                double guess = guess(trainingData.getVector());
                int actual = map(trainingData.getId());
                int error = actual - sigmoidMap(guess);
                if (error == 0) correct++;
            }
//            System.out.println("current: " + correct / trainingDataSet.size());
//            System.out.println("needed: " + percentConvergence);
        }
        return correct / trainingDataSet.size() >= percentConvergence;
    }

    public boolean timedOut(long startTime, int timeOut) {
        if ((System.currentTimeMillis() - startTime) / 1000 >= timeOut) return true;
        else return false;
    }

    private void train(Data trainingData) throws InterruptedException {
        double guess = guess(trainingData.getVector());
        int actual = map(trainingData.getId());
        double error = actual - guess;
//        weights = weights.add(trainingData.getVector().expand(power).multiplyScalar(error));
        weights = weights.add(trainingData.getVector().expand(power).multiplyScalar(error).multiplyScalar(learningRate));
        bias += error * learningRate;
    }

    private void visualTrain(Data trainingData) throws InterruptedException {
        System.out.println();
        System.out.println("------------------------");
        double guess = guess(trainingData.getVector());
        System.out.println("guess: " + guess);
        System.out.println("old bias: " + bias);
        int actual = map(trainingData.getId());
        System.out.println("id: " + trainingData.getId());
        System.out.println("mapped: " + map(trainingData.getId()));
        System.out.println("actual: " + actual);
        double error = actual - guess;
        System.out.println("error: " + actual + " - " + guess + " = " + error);
        System.out.println("learning rate: " + learningRate);
        System.out.println("training data: " + trainingData.getVector());
        System.out.println("old weights: " + weights);
        weights = weights.add(trainingData.getVector().expand(power).multiplyScalar(error).multiplyScalar(learningRate));
        System.out.println("new weights: " + weights);
        bias += error * learningRate;
        System.out.println("new bias: " + bias);
        System.out.println("------------------------");
        Thread.sleep(1000);
    }

    public double correct(ArrayList<Data> testData) {
        double correct = 0;
        for (Data data : testData) {
            double guess = sigmoidMap(guess(data.getVector()));
            int actual = this.map(data.getId());
            if (guess == actual) {
                correct++;
            }
        }
        return correct / testData.size();
    }

    public double guess(Vector input) {
//        System.out.println("input: " + input);
//        System.out.println("weights: " + weights);
//        System.out.println("input (expanded): " + input.expand(power));
        return sigmoidActivation(weights.dotProduct(input.expand(power)) + bias);
    }

    public double unactivatedGuess(Vector input) {
        return weights.dotProduct(input.expand(power)) + bias;
    }

    public double sigmoidActivation(double guess) {
        return sigmoid(guess);
    }

    public static double sigmoid(double input) { //capped to prevent overflow error, less exponential calculations
        if (input >= 13) {
            return -(1 / (1 + Math.exp(13))) + 1;
        } else if (input <= -13) {
            return -(1 / (1 + Math.exp(-13))) + 1;
        }
        return -(1 / (1 + Math.exp(input))) + 1;
    }

    public static double sigmoidDerivative(double input) {
        return sigmoid(input) * (1 - sigmoid(input));
    }

    private int sigmoidMap(double error) {
        if (error >= 0.5) return 1; else return 0;
    }

    public int map(String id) {
        if (id.equals(classifyFor)) return 1;
        return 0;
    }

    public Vector getWeights() {
        return this.weights;
    }

    public void shiftWeight(int weight, double shift) {
        weights.set(weight, weights.get(weight) + shift);
    }

    public void updateWeight(int weight, double newWeight) {
        weights.set(weight, newWeight);
    }

    public double getBias() {
        return bias;
    }

    public void updateBias(double newBias) {
        this.bias = newBias;
    }

    public String equation() {
        String out = "";
        for (int i = 0; i < dataLength; i++) {
            for (int pow = 1; pow <= power; pow++) {
                System.out.print(weights.get(i * power + pow - 1) + visualizableDimensions[i] + "^" + pow + " + ");
            }
        }
        System.out.println(bias + " = 0");
        return out;
    }

    //VISUALIZATIONS AND TESTING
    public void visualTrain(ArrayList<Data> trainingDataSet, int numDataSets) throws InterruptedException {
        Graph graph = new Graph(trainingDataSet, numDataSets);
        Graph.displayChart(graph.getChart());
        for (int rep = 0; rep < epochs; rep++) {
            Collections.shuffle(trainingDataSet);
            for (Data trainingData : trainingDataSet) {
                graph.mark(trainingData.getVector().get(0), trainingData.getVector().get(1));
                train(trainingData);
                graph.drawLine(graph.generateLine(weights, bias));
                Thread.sleep(1000);
                graph.clearLine();
                graph.removeMarker();
            }
        }
    }

}
