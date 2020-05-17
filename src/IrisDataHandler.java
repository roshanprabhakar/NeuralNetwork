import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;

public class IrisDataHandler {

    private ArrayList<String> lines;

    public IrisDataHandler(String filepath) {
        this.lines = getLines(filepath);
    }

    public ArrayList<IrisData> getData() {
        Vector[] input = getInputData();
        Vector[] output = getOutputData();
        ArrayList<IrisData> out = new ArrayList<>();
        for (int i = 0; i < input.length; i++) {
            out.add(new IrisData(input[i], output[i]));
        }
        return out;
    }

    public Vector[] getInputData() {
        Vector[] inputData = new Vector[lines.size()];
        for (int i = 0; i < lines.size(); i++) {
            String[] dataPoint = lines.get(i).split(",");
            double[] input = new double[4];
            for (int j = 0; j < 4; j++) {
                input[j] = Double.parseDouble(dataPoint[j]);
            }
            inputData[i] = new Vector(input);
        }
        return inputData;
    }

    public Vector[] getOutputData() {
        Vector[] outputData = new Vector[lines.size()];
        for (int i = 0; i < lines.size(); i++) {
            String[] dataPoint = lines.get(i).split(",");
            double[] output = new double[1];
            if (dataPoint[4].equals("Iris-setosa")) {
                output[0] = 0;
            } else if (dataPoint[4].equals("Iris-virginica")) {
                output[0] = 1;
            } else if (dataPoint[4].equals("Iris-versicolor")) {
                output[0] = 2;
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
