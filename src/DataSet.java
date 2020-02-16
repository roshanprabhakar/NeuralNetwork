import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;

public class DataSet {

    private ArrayList<String> lines;
    private int dataSize;
    private ArrayList<ArrayList<Integer>> possibleCombinations;

    //All possible contained data sets, one is bound to be linearly separable
    private ArrayList<ArrayList<Data>> dataSets;

    public DataSet(String filepath) {
        this.lines = read(filepath);
        dataSize = dataSize(lines);
        int[] indices = indices(dataSize);
        possibleCombinations = new ArrayList<>();
        for (int i = 1; i <= dataSize; i++) {
            findCombinations(indices, indices.length, i);
        }
        dataSets = new ArrayList<>();
        for (int i = 0; i < possibleCombinations.size(); i++) {
            dataSets.add(getData(possibleCombinations.get(i)));
        }
    }

    private void combinationUtil(int[] arr, int[] data, int start, int end, int index, int r) {
        if (index == r) {
            ArrayList<Integer> row = new ArrayList<>();
            for (int j = 0; j < r; j++) {
                row.add(data[j]);
            }
            possibleCombinations.add(row);
            return;
        }
        for (int i = start; i <= end && end - i + 1 >= r - index; i++) {
            data[index] = arr[i];
            combinationUtil(arr, data, i + 1, end, index + 1, r);
        }
    }

    private void findCombinations(int[] arr, int n, int r) {
        int[] data = new int[r];
        combinationUtil(arr, data, 0, n - 1, 0, r);
    }

    private int dataSize(ArrayList<String> lines) {
        int minDataSize = Integer.MAX_VALUE;
        for (String line : lines) {
            int dataSize = dataSize(line);
            if (dataSize < minDataSize) {
                minDataSize = dataSize;
            }
        }
        return minDataSize;
    }

    private int dataSize(String line) {
        String[] data = line.split(",");
        int dataSize = 0;
        for (int i = 0; i < data.length; i++) {
            try {
                Double.parseDouble(data[i]);
            } catch (NumberFormatException e) {
                dataSize = i;
                break;
            }
        }
        return dataSize;
    }

    private int[] indices(int dataSize) {
        int[] out = new int[dataSize];
        for (int i = 0; i < dataSize; i++) {
            out[i] = i;
        }
        return out;
    }

    public int getDataSize() {
        return this.dataSize;
    }

    public ArrayList<ArrayList<Data>> getDataSets() {
        return dataSets;
    }

    public ArrayList<Data> getData(int dataLength, int starti) {
        return getData(lines, dataLength, starti);
    }

    public ArrayList<Data> getData(ArrayList<Integer> indices) {
        return getData(lines, indices);
    }

    public static ArrayList<Data> getData(ArrayList<String> lines, int dataLength, int starti) {
        ArrayList<Data> data = new ArrayList<>();
        for (String line : lines) {
            data.add(getData(line, dataLength, starti));
        }
        Collections.shuffle(data);
        return data;
    }

    public static ArrayList<Data> getData(ArrayList<String> lines, ArrayList<Integer> indices) {
        ArrayList<Data> data = new ArrayList<>();
        for (String line : lines) {
            data.add(getData(line, indices));
        }
        Collections.shuffle(data);
        return data;
    }

    public static Data getData(String line, int dataLength, int starti) {

        String[] items = line.split(",");
        double[] data = new double[dataLength];
        for (int i = 0; i < dataLength; i++) {
            data[i] = Double.parseDouble(items[i + starti]);
        }
        return new Data(items[items.length - 1], new Vector(data));
    }

    public static Data getData(String line, ArrayList<Integer> indices) {

        String[] items = line.split(",");
        double[] data = new double[indices.size()];
        for (int i = 0; i < indices.size(); i++) {
            data[i] = Double.parseDouble(items[indices.get(i)]);
        }
        return new Data(items[items.length - 1], new Vector(data));
    }

    public ArrayList<ArrayList<Integer>> getPossibleCombinations() {
        return possibleCombinations;
    }

    public static ArrayList<String> read(String filepath) {
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

    public static void printSet(ArrayList<Data> dataSet) {
        String oldTitle = "";
        for (Data data : dataSet) {
            if (!oldTitle.equals(data.getId())) {
                oldTitle = data.getId();
                System.out.println(oldTitle);
            }
            for (int i = 0; i < data.size(); i++) {
                System.out.print(data.getVector().get(i));
                if (i < data.size() - 1) System.out.print(", ");
            }
            System.out.println();
        }
    }
}
