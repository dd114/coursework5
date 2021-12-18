package huch.company;

public class Main {
    public static void main(String[] args) {
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
}




//http://robocraft.ru/blog/algorithm/560.html
//https://habr.com/ru/post/313216/
