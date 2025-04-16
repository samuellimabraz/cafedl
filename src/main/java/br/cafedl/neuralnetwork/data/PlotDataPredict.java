package br.cafedl.neuralnetwork.data;

import org.math.plot.Plot2DPanel;
import org.math.plot.Plot3DPanel;
import org.nd4j.linalg.api.ndarray.INDArray;

import javax.swing.*;
import java.awt.*;

public class PlotDataPredict {
    public void plot2d(INDArray x, INDArray y, INDArray predict, String title) {
        // Criação do gráfico 2D
        double[] xData = x.data().asDouble();
        double[] yData = y.data().asDouble();
        double[] predictData = predict.data().asDouble();

        // create your PlotPanel (you can use it as a JPanel) with a legend at SOUTH
        Plot2DPanel plot = new Plot2DPanel();
        plot.addLegend("SOUTH");

        // add grid plot to the PlotPanel
        plot.addScatterPlot("Dados", xData, yData);
        plot.addScatterPlot("Modelo", xData, predictData);

        // put the PlotPanel in a JFrame like a JPanel
        JFrame frame = new JFrame(title);
        frame.setSize(800, 600);
        frame.setContentPane(plot);
        frame.setVisible(true);
    }

    public void plot3dGridandScatter(INDArray x, INDArray y, INDArray predict, String title) {
        Plot3DPanel plot = new Plot3DPanel("SOUTH");

        // Create 3d graph with scatter data and predict
        plot.addScatterPlot("Dados", x.getColumn(0).toDoubleVector(), x.getColumn(1).toDoubleVector(), y.toDoubleVector());
        plot.addScatterPlot("Modelo", x.getColumn(0).toDoubleVector(), x.getColumn(1).toDoubleVector(), predict.toDoubleVector());


        // put the PlotPanel in a JFrame like a JPanel
        JFrame frame = new JFrame(title);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(800, 600);
        frame.setLayout(new BorderLayout());
        frame.add(plot, BorderLayout.CENTER);
        frame.setVisible(true);
    }
}

