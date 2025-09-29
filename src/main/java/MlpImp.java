import java.util.Random;

public class MlpImp {
    /*
    Pesos
     */
    private  double [][] Wh;
    private double [][] Wtheta;

    private static final double learningRate = 0.3;

    private double[][] getWtheta() {return Wtheta;}

    private void setWtheta(double[][] wtheta) {Wtheta = wtheta;}

    private double[][] getWh() {return Wh;}

    private void setWh(double[][] wh) {Wh = wh;}

    public MlpImp(int qtdaIn, int qtdaOut, int qtdaHidden) {
        setWh(new double[qtdaIn + 1][qtdaHidden]);
        setWtheta(new double[qtdaHidden + 1][qtdaOut]);

        Random random = new Random();
        for (int i = 0; i < getWh().length; i++) {
            for (int j = 0; j < getWh()[i].length; j++) {
                getWh()[i][j] = (random.nextDouble() * 0.06) - 0.03;
            }
        }
        for (int i = 0; i < getWtheta().length; i++) {
            for (int j = 0; j < getWtheta()[i].length; j++) {
                getWtheta()[i][j] = (random.nextDouble() * 0.06) - 0.03;
            }
        }
    }

    private static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    private static double sigmoidFromOutput(double y) {
        return y * (1.0 - y);
    }

    public double[] train(double[] input, double[] target) {
        int nIn = getWh().length - 1;
        int nHidden = getWh()[0].length;
        int nOut = getWtheta()[0].length;

        double[] inBias = new double[nIn + 1];
        for (int i = 0; i < nIn; i++) {
            inBias[i] = input[i];
        }
        inBias[nIn] = 1.0;

        double[] hidden = new double[nHidden];
        for (int i = 0; i < nHidden; i++) {
            double sum = 0.0;
            for (int j = 0; j < nIn + 1; j++) {
                sum += inBias[j] * getWh()[j][i];
            }
            hidden[i] = sigmoid(sum);
        }

        double[] hiddenBias = new double[nHidden + 1];
        for (int i = 0; i < nHidden; i++) {
            hiddenBias[i] = hidden[i];
        }
        hiddenBias[nHidden] = 1.0;

        double[] output = new double[nOut];
        for (int i = 0; i < nOut; i++) {
            double sum = 0.0;
            for (int j = 0; j < nHidden + 1; j++) {
                sum += hiddenBias[j] * getWtheta()[j][i];
            }
            output[i] = sigmoid(sum);
        }

        double[] deltaOut = new double[nOut];
        for (int i = 0; i < nOut; i++) {
            double error = target[i] - output[i];
            deltaOut[i] = error * sigmoidFromOutput(output[i]);
        }

        double[] deltaHidden = new double[nHidden];
        for (int i = 0; i < nHidden; i++) {
            double sum = 0.0;
            for (int j = 0; j < nOut; j++) {
                sum += deltaOut[j] * getWtheta()[i][j];
            }
            deltaHidden[i] = sum * sigmoidFromOutput(hidden[i]);
        }

        for (int i = 0; i < nHidden + 1; i++) {
            for (int j = 0; j < nOut; j++) {
                getWtheta()[i][j] += learningRate * hiddenBias[i] * deltaOut[j];
            }
        }

        for (int i = 0; i < nIn + 1; i++) {
            for (int j = 0; j < nHidden; j++) {
                getWh()[i][j] += learningRate * inBias[i] * deltaHidden[j];
            }
        }

        return output;
    }
}
