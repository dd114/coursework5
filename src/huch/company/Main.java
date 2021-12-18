package huch.company;

import huch.MNIST.MnistDataReader;
import huch.MNIST.MnistMatrix;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class Main {
    public static void main(String[] args) throws IOException {

        MnistMatrix[] mnistMatrixTrain = new MnistDataReader().readData("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", false);
//        System.out.println(mnistMatrixTrain.length);
//        printMnistMatrix(mnistMatrixTrain[mnistMatrixTrain.length - 1]);

        MnistMatrix[] mnistMatrixTest = new MnistDataReader().readData("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte", false);
//        System.out.println(mnistMatrixTest.length);
//        printMnistMatrix(mnistMatrixTest[0]);






        //NeuralNetwork(кол-во слоев и нейронов в них, [скорость обучения], [продуктивность])
        MNN network = new MNN(new int[][]{{28 * 28}, {16}, {16}, {10}}, 0.1f, 0.3f);

//        network.load("save.txt");

        int epochs = 1;

        for(int i = 0; i < epochs; i++) {

            for (int j = 0; j < 60000; j++) {

                float[] trainInput = new float[28 * 28];
                straightenArray(trainInput, mnistMatrixTrain[j]);

                int[] trainOutput = new int[10];
                trainOutput[mnistMatrixTrain[j].getLabel()] = 1;

//                for (int k = 0; k < 10; k++)
//                    System.out.print(trainOutput[k] + " ");
//                System.out.println();

                network.train(trainInput, trainOutput);
            }

//            network.error("error.txt");
        }


        // input value for predict
        MnistMatrix testDigit = mnistMatrixTrain[11];
        float[] testInput = new float[28 * 28];
        straightenArray(testInput, testDigit);
        //printArray(testInput);

        System.out.println("Predict number = " + network.predict(testInput));
        System.out.println("Precise number = " + testDigit.getLabel());

        //printMnistMatrix(testDigit);


        printPicture(testDigit, "digit.jpg");

        network.save("save.txt");
        //System.out.println(network.predict(new float[]{1, 1, 1}));

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

    private static void straightenArray(float[] array, final MnistMatrix matrix) {
        for (int r = 0; r < matrix.getNumberOfRows(); r++ ) {
            for (int c = 0; c < matrix.getNumberOfColumns(); c++) {
                array[r * 28 + c] = matrix.getValue(r, c);
            }
        }
    }

    private static void printArray(float[] array){
        for (int r = 0; r < 28; r++ ) {
            for (int c = 0; c < 28; c++) {
                System.out.print(array[28 * r + c] + " ");
            }
            System.out.println();
        }
    }

    private static void printPicture(final MnistMatrix matrix, String path){
        try {

            // Создаем новое пустое изображение, такого же размера
            BufferedImage result = new BufferedImage(matrix.getNumberOfRows(), matrix.getNumberOfColumns(), BufferedImage.TYPE_INT_RGB);

            // Делаем двойной цикл, чтобы обработать каждый пиксель
            for (int x = 0; x < matrix.getNumberOfRows(); x++) {
                for (int y = 0; y < matrix.getNumberOfColumns(); y++) {


                    // Применяем стандартный алгоритм для получения черно-белого изображения
                    int grey = matrix.getValue(x, y);

                    // Если вы понаблюдаете, то заметите что у любого оттенка серого цвета, все каналы имеют
                    // одно и то же значение. Так, как у нас изображение тоже будет состоять из оттенков серого
                    // то, все канали будут иметь одно и то же значение.
                    int newRed = grey;
                    int newGreen = grey;
                    int newBlue = grey;

                    //  Cоздаем новый цвет
                    Color newColor = new Color(newRed, newGreen, newBlue);

                    // И устанавливаем этот цвет в текущий пиксель результирующего изображения
                    // (y, x), тк иначе получается перевернутое изображение хз почему
                    result.setRGB(y, x, newColor.getRGB());
                }
            }

            // Созраняем результат в новый файл
            File output = new File(path);
            ImageIO.write(result, "jpg", output);

        } catch (IOException e) {

            // При открытии и сохранении файлов, может произойти неожиданный случай.
            // И на этот случай у нас try catch
            System.out.println("Файл не найден или не удалось сохранить");
        }
    }

}




//http://robocraft.ru/blog/algorithm/560.html
//https://habr.com/ru/post/313216/
