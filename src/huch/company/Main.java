package huch.company;

public class Main {
    public static void main(String[] args) {
        //NeuralNetwork(кол-во слоев и нейронов в них, [скорость обучения], [продуктивность])
        MNN network = new MNN(new int[][]{{2}, {7}, {4}}, 0.1f, 0.3f);

//        network.load("save.txt");
        int dimX = 10000, dimY = 10000;

        int epochs = 10000;

        for(int i = 0; i < epochs; i++) {

            int x = (int) (Math.random() * dimX) - dimX / 2, y = (int) (Math.random() * dimY) - dimY / 2;
            float[] input = new float[2];
            input[0] = x;
            input[1] = y;

            int[] output = new int[4];
            int index = -1;

            if(input[0] >= 0 && input[1] >= 0) {
                index = 0;
                output[0] = 1;
            }
            if(input[0] < 0 && input[1] >= 0) {
                index = 1;
                output[1] = 1;
            }
            if(input[0] < 0 && input[1] < 0) {
                index = 2;
                output[2] = 1;
            }
            if(input[0] >= 0 && input[1] < 0) {
                index = 3;
                output[3] = 1;
            }

//            System.out.println("x = " + input[0] + " y = " + input[1]);
//            System.out.println("index = " + index);

            network.train(input, output);

            network.error("error.txt");
        }

        network.save("save.txt");

        int error = 0;
        int test = 10000;
        for (int i = 0; i < test; i++) {


            int x = (int) (Math.random() * dimX) - dimX / 2, y = (int) (Math.random() * dimY) - dimY / 2;
            float[] input = new float[2];
            input[0] = x;
            input[1] = y;

            int[] output = new int[4];
            int index = -1;

            if (input[0] >= 0 && input[1] >= 0) {
                index = 0;
                output[0] = 1;
            }
            if (input[0] < 0 && input[1] >= 0) {
                index = 1;
                output[1] = 1;
            }
            if (input[0] < 0 && input[1] < 0) {
                index = 2;
                output[2] = 1;
            }
            if (input[0] >= 0 && input[1] < 0) {
                index = 3;
                output[3] = 1;
            }

//            System.out.println("x = " + input[0] + " y = " + input[1]);
//            System.out.println("index = " + index);
            int predict = network.predict(input);
            if(predict != index) error++;
//            System.out.println(predict);

        }


        System.out.println("My test = " + network.predict(new float[]{1, -1}));
        System.out.println("error = " + error);
        System.out.println("percent of error = " + ((float) error / test) * 100 + " %");








//        System.out.println(network.predict(new float[]{0, 0, 1}));
//        System.out.println(network.predict(new float[]{0, 1, 0}));
//        System.out.println(network.predict(new float[]{0, 1, 1}));
//        System.out.println(network.predict(new float[]{1, 0, 0}));
//        System.out.println(network.predict(new float[]{1, 0, 1}));
//        System.out.println(network.predict(new float[]{1, 1, 0}));
//        System.out.println(network.predict(new float[]{1, 1, 1}));

    }
}




//http://robocraft.ru/blog/algorithm/560.html
//https://habr.com/ru/post/313216/
