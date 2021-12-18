package huch.company;

import huch.MNIST.MnistDataReader;
import huch.MNIST.MnistMatrix;

import java.io.IOException;

public class Main {
    public static void main(String[] args) throws IOException {

        MnistMatrix[] mnistMatrix = new MnistDataReader().readData("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte");
        printMnistMatrix(mnistMatrix[mnistMatrix.length - 1]);
        mnistMatrix = new MnistDataReader().readData("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte");
        printMnistMatrix(mnistMatrix[0]);






        //NeuralNetwork(кол-во слоев и нейронов в них, [скорость обучения], [продуктивность])
        MNN network = new MNN(new int[][]{{2}, {7}, {4}}, 0.1f, 0.3f);

//        network.load("save.txt");
        int dimX = 10000, dimY = 10000;

        int epochs = 10000;

        for(int i = 0; i < epochs; i++) {


            network.train(new float[]{1, 1}, new int[]{0, 1, 0, 0});

            network.error("error.txt");
        }

        network.save("save.txt");


//        System.out.println(network.predict(new float[]{1, 1, 1}));

    }


    private static void printMnistMatrix(final MnistMatrix matrix) {
        System.out.println("label: " + matrix.getLabel());
        for (int r = 0; r < matrix.getNumberOfRows(); r++ ) {
            for (int c = 0; c < matrix.getNumberOfColumns(); c++) {
                System.out.print(matrix.getValue(r, c) + " ");
            }
            System.out.println();
        }
    }
}




//http://robocraft.ru/blog/algorithm/560.html
//https://habr.com/ru/post/313216/
