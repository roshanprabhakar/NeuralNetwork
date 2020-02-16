import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.annotations.XYAnnotation;
import org.jfree.chart.plot.Plot;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYItemRenderer;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.general.Series;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.util.ShapeUtilities;

import javax.swing.*;
import java.awt.*;
import java.time.temporal.ChronoField;
import java.util.ArrayList;

public class Graph {

    private ArrayList<XYSeries> series;
    private XYSeriesCollection collection;
    private int numDataGroups;

    private JFreeChart chart;

    public Graph(ArrayList<Data> dataSet, int numDataGroups) {

        series = new ArrayList<>();
        String oldID = dataSet.get(0).getId();
        XYSeries series = new XYSeries(dataSet.get(0).getId());

        for (Data data : dataSet) {
            if (!data.getId().equals(oldID)) {
                oldID = data.getId();
                this.series.add(series);
                series = new XYSeries(data.getId());
            }
            series.add(data.getVector().get(0), data.getVector().get(1));
        }
        this.series.add(series);

        XYSeriesCollection xySeriesCollection = new XYSeriesCollection();
        for (XYSeries xySeries : this.series) {
            xySeriesCollection.addSeries(xySeries);
        }

        collection = xySeriesCollection;
        chart = createChart(collection);

        XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
        for (int i = 0; i < numDataGroups; i++) {
            renderer.setSeriesLinesVisible(i, false);
        }

        XYPlot plot = chart.getXYPlot();
        plot.setRenderer(renderer);
        this.numDataGroups = numDataGroups;
    }

    public Graph() {
        series = new ArrayList<>();
        collection = new XYSeriesCollection();
    }

    public static JFreeChart createChart(XYSeriesCollection collection) {
        return ChartFactory.createXYLineChart(
                "Chart" ,
                "Category" ,
                "Output" ,
                collection,
                PlotOrientation.VERTICAL ,
                true , true , false);
    }

    public XYSeries generateLine(Vector weights, double bias) { //assumes a line, not curve
        XYSeries line = new XYSeries("regression result");
        double highestY = chart.getXYPlot().getRangeAxis().getUpperBound();
        double lowestY = chart.getXYPlot().getRangeAxis().getLowerBound();
        double highestX = chart.getXYPlot().getDomainAxis().getUpperBound();
        double lowestX = chart.getXYPlot().getDomainAxis().getLowerBound();


        double output;
        for (double x = lowestX; x < highestX; x += 0.1) {
            output = output(weights.get(0), weights.get(1), bias, x);
//            System.out.println(x + ", " + output);
            if (output < highestY && output > lowestY) {
                line.add(x, output);
            }
        }

        return line;
    }

    public void addLine(XYSeries series) {
        collection.addSeries(series);
    }

    public void addPoint(double x, double y) {
        collection.getSeries(0).add(x, y);
    }

    public void drawLine(XYSeries line) {
        this.collection.addSeries(line);
    }

    public void clearLine() {
        collection.removeSeries(numDataGroups);
    }

    private double output(double xcoeff, double ycoeff, double bias, double input) {
//        ay + bx + bias = 0;
//        ay =  -bx - bias;
//        y = -(bx / a) - (bias / a);
        return -1 * (xcoeff * input / ycoeff) - (bias / ycoeff);
    }

    public void mark(double x, double y) {
        XYSeries series = new XYSeries("Marker");
        series.add(x, y);
        collection.addSeries(series);
        Shape cross = ShapeUtilities.createDiagonalCross(3, 10);
        XYPlot plot = (XYPlot) chart.getPlot();
        XYItemRenderer renderer = plot.getRenderer();
        renderer.setSeriesShape(numDataGroups, cross);
    }

    public void removeMarker() {
        collection.removeSeries(numDataGroups);
    }

    public static void displayChart(JFreeChart chart) {
        ChartPanel panel = new ChartPanel(chart);
        JFrame frame = new JFrame("Chart Frame");
        frame.getContentPane().add(panel);
        frame.pack();
        frame.setVisible(true);
    }

    public XYSeriesCollection getCollection() {return collection;}

    public JFreeChart getChart() {return chart;}
}
