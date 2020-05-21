import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;

public class IrisDataHandler {

    private ArrayList<String> lines;

    public IrisDataHandler(String filepath) {
        this.lines = getLines(filepath);
    }

    public HashMap<String, ArrayList<NetworkData>> getData(double percentTraining) {
        Vector[] input = getInputData();
        Vector[] output = getOutputData();
        ArrayList<NetworkData> out = new ArrayList<>();
        for (int i = 0; i < input.length; i++) {
            out.add(new NetworkData(input[i], output[i]));
        }
        Collections.shuffle(out);
        HashMap<String, ArrayList<NetworkData>> split = new HashMap<>();
        ArrayList<NetworkData> training = new ArrayList<>();
        ArrayList<NetworkData> testing = new ArrayList<>();
        for (int i = 0; i < out.size(); i++) {
            if (i < out.size() * percentTraining) training.add(out.get(i));
            else testing.add(out.get(i));
        }
        Collections.shuffle(training);
        Collections.shuffle(testing);
        split.put("training", training);
        split.put("testing", testing);
        return split;
    }

    public Vector[] getInputData() {
        Vector[] inputData = new Vector[lines.size()];
        for (int i = 0; i < lines.size(); i++) {
            String[] dataPoint = lines.get(i).split(",");
            double[] input = new double[4];
            for (int j = 0; j < 4; j++) {
                if (j == 0) {
                    input[j] = 2.8 * Perceptron.sigmoid(Double.parseDouble(dataPoint[j]) - 5);
                } else if (j == 1) {
                    input[j] = 2.8 * Perceptron.sigmoid(Double.parseDouble(dataPoint[j]) - 2.5);
                } else if (j == 2) {
                    input[j] = 2.8 * Perceptron.sigmoid(Double.parseDouble(dataPoint[j]) - 3.5);
                } else {
                    input[j] = 2.8 * Perceptron.sigmoid(3.5 * Double.parseDouble(dataPoint[j]) - 4);
                }
//                input[j] = Double.parseDouble(dataPoint[j]);
            }
            inputData[i] = new Vector(input);
        }
        return inputData;
    }

    public Vector[] getOutputData() {
        Vector[] outputData = new Vector[lines.size()];
        for (int i = 0; i < lines.size(); i++) {
            String[] dataPoint = lines.get(i).split(",");
            double[] output = new double[3];
            if (dataPoint[4].equals("Iris-setosa")) {
//                output = new double[]{1,0,0,0,0,0,0,0,0};
                output = new double[]{1,0,0};
            } else if (dataPoint[4].equals("Iris-virginica")) {
//                output = new double[]{0,0,0,0,1,0,0,0,0};
                output = new double[]{0,1,0};
            } else if (dataPoint[4].equals("Iris-versicolor")) {
//                output = new double[]{0,0,0,0,0,0,0,0,1};
                output = new double[]{0,0,1};
            }
            outputData[i] = new Vector(output);
        }
        return outputData;
    }

    public static ArrayList<String> getLines(String filepath) {
        ArrayList<String> lines = new ArrayList<>();
        try {
            BufferedReader reader = new BufferedReader(new FileReader(new File(filepath)));
            String line;
            while ((line = reader.readLine()) != null) {
                lines.add(line);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return lines;
    }
}
