import converter.IrisConverter;
import converter.Register;
import converter.VertebralColumnConverter;

import java.util.List;

public class MlpMain {

    public static void main(String[] args) {

        /*
        Controlador de base
         */
        boolean isVertebralColumn = true;

        VertebralColumnConverter vertebralColumnConverter = new VertebralColumnConverter();
        IrisConverter irisConverter = new IrisConverter();

        if (isVertebralColumn) {
            List<Register> dataFromBase = vertebralColumnConverter.trainConverter();
            MlpImp mlpAnd = new MlpImp(2, 1, 2);

            int total = dataFromBase.size();

            for (int epoch = 0; epoch < 100000; epoch++) {
                double approxErrTraining = 0.0;
                int errClsTraining = 0;

                System.out.println("=== TREINO ===");
                for (Register register : dataFromBase) {
                    double[] out = mlpAnd.train(register.getFeatures(), register.getTarget());
                    approxErrTraining += Math.abs(out[0] - register.getTarget()[0]);
                    int pred = out[0] >= 0.5 ? 1 : 0;
                    int tgt = (int) register.getTarget()[0];
                    if (pred != tgt) errClsTraining++;
                }
                double taxaErroClss = (double) errClsTraining / total;
                System.out.printf("[TRAIN VERTEBRAL COLUMN] Epoch: %d - ApproxError: %.4f - ClassError: %d/%d (%.2f%%)%n",
                        epoch, approxErrTraining, errClsTraining, total, taxaErroClss * 100.0);

                double approxErrTest = 0.0;
                int errClsTest = 0;

                List<Register> dataFromBaseTest = vertebralColumnConverter.testConverter();
                System.out.println("=== TESTE ===");
                for (Register register : dataFromBaseTest) {
                    double[] out = mlpAnd.test(register.getFeatures(), register.getTarget());
                    approxErrTest += Math.abs(out[0] - register.getTarget()[0]);
                    int pred = out[0] >= 0.5 ? 1 : 0;
                    int tgt = (int) register.getTarget()[0];
                    if (pred != tgt) errClsTest++;
                }
                int totalTest = dataFromBaseTest.size();
                double taxaErroClassTest = (double) errClsTest / totalTest;
                System.out.printf("[TEST VERTEBRAL COLUMN] Epoch: %d - ApproxError: %.4f - ClassError: %d/%d (%.2f%%)%n",
                        epoch, approxErrTest, errClsTest, totalTest, taxaErroClassTest * 100.0);
                System.out.println("\n");
            }
        } else {
            List<Register> dataFromBase = irisConverter.trainConverter();
            MlpImp mlpAnd = new MlpImp(2, 1, 2);

            int total = dataFromBase.size();

            for (int epoch = 0; epoch < 10000; epoch++) {
                double approxErrTraining = 0.0;
                int errClsTraining = 0;

                System.out.println("=== TREINO ===");
                for (Register register : dataFromBase) {
                    double[] out = mlpAnd.train(register.getFeatures(), register.getTarget());
                    approxErrTraining += Math.abs(out[0] - register.getTarget()[0]);
                    int pred = out[0] >= 0.5 ? 1 : 0;
                    int tgt = (int) register.getTarget()[0];
                    if (pred != tgt) errClsTraining++;
                }
                double taxaErroClss = (double) errClsTraining / total;
                System.out.printf("[TRAIN VERTEBRAL COLUMN] Epoch: %d - ApproxError: %.4f - ClassError: %d/%d (%.2f%%)%n",
                        epoch, approxErrTraining, errClsTraining, total, taxaErroClss * 100.0);

                double approxErrTest = 0.0;
                int errClsTest = 0;

                List<Register> dataFromBaseTest = irisConverter.testConverter();
                System.out.println("=== TESTE ===");
                for (Register register : dataFromBaseTest) {
                    double[] out = mlpAnd.test(register.getFeatures(), register.getTarget());
                    approxErrTest += Math.abs(out[0] - register.getTarget()[0]);
                    int pred = out[0] >= 0.5 ? 1 : 0;
                    int tgt = (int) register.getTarget()[0];
                    if (pred != tgt) errClsTest++;
                }
                int totalTest = dataFromBaseTest.size();
                double taxaErroClssTest = (double) errClsTest / totalTest;
                System.out.printf("[TEST VERTEBRAL COLUMN] Epoch: %d - ApproxError: %.4f - ClassError: %d/%d (%.2f%%)%n",
                        epoch, approxErrTest, errClsTest, totalTest, taxaErroClssTest * 100.0);
                System.out.println("\n");
            }

        }
    }
}

