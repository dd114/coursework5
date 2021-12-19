package huch.company;

import java.io.*;

public class MNN {
    private float learning_rate = 0.01f;
    private float moment = 0.3f;

    public float[][][] overallLayer;
    //private float[][][] deltaOverallLayer;
    private float[][] error_neuron;

    public MNN(int[][] layer) {
        overallLayer = new float[layer.length][0][0];
        overallLayer[0] = new float[layer[0][0]][0];

//        deltaOverallLayer = new float[layer.length][0][0];
//        deltaOverallLayer[0] = new float[layer[0][0]][0];

        for (int i = 1; i < layer.length; i++) {
            overallLayer[i] = new float[layer[i][0]][overallLayer[i - 1].length + 1];
//            deltaOverallLayer[i] = new float[layer[i][0]][overallLayer[i - 1].length + 1];
        }

        for (int i = 1; i < overallLayer.length; i++)
            for (int j = 0; j < overallLayer[i].length; j++)
                for (int k = 0; k < overallLayer[i][j].length; k++)
                    overallLayer[i][j][k] = (float) (Math.random());

    }

    public MNN(int[][] layer, float learning_rate) {
        this(layer);
        this.learning_rate = learning_rate;
    }

    public MNN(int[][] layer, float learning_rate, float moment) {
        this(layer, learning_rate);
        this.moment = moment;
    }

    public int predict(float[] inputs) {
        float[][] inputs_layer = new float[overallLayer.length][0];
        float[][] outputs_layer = new float[overallLayer.length][0];
        float max = 0;
        int answer = -1;

        outputs_layer[0] = inputs;

        for (int i = 1; i < overallLayer.length; i++) {
            inputs_layer[i] = multiplyByMatrix(overallLayer[i], outputs_layer[i - 1]);
            outputs_layer[i] = activationMapper(inputs_layer[i]);
        }

//        for (float b : outputs_layer[outputs_layer.length - 1])  System.out.println(b); // выводим сигнал каждого выхода (результат каждого аутпута)

        for (int i = 0; i < outputs_layer[outputs_layer.length - 1].length - 1; i++) {
            System.out.println(i + " = " + outputs_layer[outputs_layer.length - 1][i]);
        }

        // находим максимальный из них, который и будет являться ответом
        for (int i = 0; i < outputs_layer[outputs_layer.length - 1].length - 1; i++)
            if (outputs_layer[outputs_layer.length - 1][i] > max) {
                max = outputs_layer[outputs_layer.length - 1][i];
                answer = i;
            }
        return answer; //выводим номер нейрона соответствующему определенному ответу
    }

    public void train(float[] inputs, int[] correct_predict) {
        float[][] inputs_layer = new float[overallLayer.length][0];
        float[][] outputs_layer = new float[overallLayer.length][0];
        error_neuron = new float[overallLayer.length][0];
        outputs_layer[0] = inputs;

        for (int i = 1; i < overallLayer.length; i++) {
            inputs_layer[i] = multiplyByMatrix(overallLayer[i], outputs_layer[i - 1]);
            outputs_layer[i] = activationMapper(inputs_layer[i]);
        }

        int last_layer = overallLayer.length - 1;


        for (int i = 0; i < last_layer; i++)
            error_neuron[i] = new float[overallLayer[i].length + 1]; //заполняем массив количеством нейронов, +1 - нейрон смещения
        error_neuron[last_layer] = new float[overallLayer[last_layer].length]; //заполняем последний слой массива без нейрона смещения

        //считаем ошибку всех нейронов на последнем скрытом слое
        for (int i = 0; i < overallLayer[last_layer].length; i++) // определяем ошибку для последнего слоя по формуле correct_answer - expected_answer
            error_neuron[last_layer][i] = correct_predict[i] - outputs_layer[last_layer][i];

        // тк ошибки нет на первом входном слое => i >= 1, а i = last_layer - 1, тк последний слой заполнили выше, ошибка также расставлется и на нейрон смещения
        for (int i = last_layer - 1; i >= 1; i--)
            for (int j = 0; j < error_neuron[i].length; j++) { // смотрим каждый нейрон слоя i
                for (int k = 0; k < overallLayer[i + 1].length; k++) { // смотрим каждый нейрон слоя i + 1
                    //метод обратного распостранения ошибки, умножаем ошибку на вес
                    error_neuron[i][j] += error_neuron[i + 1][k] * overallLayer[i + 1][k][j];
                }
                // затем сумму умножаем на производную функции активации и получаем итоговую ошибку
                error_neuron[i][j] = error_neuron[i][j] * derivativeActivation(inputs_layer[i][j]);
            }


        for (int i = 1; i < overallLayer.length; i++) // добавить нейрон смещения и дельту весов
            for (int j = 0; j < overallLayer[i].length; j++) {
                for (int k = 0; k < overallLayer[i][j].length - 1; k++) {
                    //по формуле произведения производной сигмоиды, на выход предыдущего слоя и ошибки плюс изменение весов (дельта весов) на предыдущей итерации помноженое на момент
                    overallLayer[i][j][k] -= learning_rate * error_neuron[i][j] * outputs_layer[i - 1][k] * derivativeActivation(inputs_layer[i][j]);
                    //запоминаем их тут, чтобы использовать на следующей итерации
//                    deltaOverallLayer[i][j][k] = learning_rate * error_neuron[i][j] * outputs_layer[i][j] * (1 - outputs_layer[i][j]) * outputs_layer[i - 1][k];
                }
                // корректируем весы нейрона смещения
                overallLayer[i][j][overallLayer[i][j].length - 1] -= learning_rate * error_neuron[i][j] * 1 * derivativeActivation(inputs_layer[i][j]);
                //запоминаем их тут, чтобы использовать на следующей итерации
//                deltaOverallLayer[i][j][deltaOverallLayer[i][j].length - 1] = learning_rate * error_neuron[i][j] * outputs_layer[i][j] * (1 - outputs_layer[i][j]) * 1;
            }
    }

    private float[] sigmoidMapper(float[] inputs) {
        float[] outputs = new float[inputs.length];
        for (int i = 0; i < inputs.length - 1; i++) outputs[i] = sigmoid(inputs[i]);
        outputs[inputs.length - 1] = 1;
        return outputs;
    }

    private float sigmoid(float x) {
        return (float) (1 / (1 + Math.exp(-x)));
    }

    private float activation(float x){
        return sigmoid(x);
    }

    private float[] activationMapper(float[] inputs){
        return sigmoidMapper(inputs);
    }

    private float derivativeActivation(float x){
        float ePow = (float) Math.exp(-x);
        return ePow / ((ePow + 1) * (ePow + 1));
    }

    private float[] multiplyByMatrix(float[][] m1, float[] m2) { //2 version with neuron bias, 1 version above
        float[] overallM = new float[m1.length + 1];
        overallM[m1.length] = 1;
        for (int y = 0; y < m1.length; y++) {
            assert m1[y].length - 1 == m2.length;
            for (int x = 0; x < m1[y].length - 1; x++) overallM[y] += m1[y][x] * m2[x];
            overallM[y] += m1[y][m1[y].length - 1] * 1;
        }
        return overallM;
    }

    public void save(String path) {
        try {
            ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(path));
            out.writeObject(overallLayer);
            out.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void load(String path) {
        try {
            ObjectInputStream in = new ObjectInputStream(new FileInputStream(path));
            overallLayer = (float[][][]) in.readObject();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }
    }

    public void error(String path) {
        float sum_err = 0;
        try {
            FileWriter writer = new FileWriter(path, true);
            int quantity_neuron = error_neuron[error_neuron.length - 1].length;
            for (int i = 0; i < quantity_neuron; i++) sum_err += Math.pow(error_neuron[error_neuron.length - 1][i], 2);
            writer.write(sum_err / quantity_neuron + "\n");
            writer.flush();
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}

//                    System.out.println("Ошибка нейрона " + i + " " + j + " = " + error_neuron[i + 1][k] + " * " + overallLayer[i + 1][k][j] + " = " + error_neuron[i + 1][k] * overallLayer[i + 1][k][j]);
//                    System.out.println(i + " " + j + " " + k + " Текущий вес " + overallLayer[i][j][k] + " Ошибка = " + error_neuron[i][j] + " Вход нейрона = " + inputs_layer[i][j] + " Выход нейрона = " + outputs_layer[i][j] + " Предыдущий выход нейрона = " + outputs_layer[i - 1][j] + " Изменение на " + learning_rate * outputs_layer[i][j] * (1 - outputs_layer[i][j]) * error_neuron[i][j]);
