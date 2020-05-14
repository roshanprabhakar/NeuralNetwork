import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;

public class IrisDataHandler {

    private ArrayList<String> lines;
    private String target;

    public IrisDataHandler(String target, String filepath) {
        this.lines = getLines(filepath);
        this.target = target;
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
            if (dataPoint[4].equals(target)) {
                output[0] = 1;
            } else output[0] = 0;
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
