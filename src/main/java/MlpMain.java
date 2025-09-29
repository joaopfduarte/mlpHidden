public class MlpMain {

    private static final double[][][] baseI = {
            {{0, 0}, {0}},
            {{0, 1}, {0}},
            {{1, 0}, {0}},
            {{1, 1}, {1}}
    };

    private static final double[][][] baseII = {
            {{0, 0}, {0}},
            {{0, 1}, {1}},
            {{1, 0}, {1}},
            {{1, 1}, {1}}
    };

    private static final double[][][] baseIII = {
            {{0, 0}, {0}},
            {{0, 1}, {1}},
            {{1, 0}, {1}},
            {{1, 1}, {0}}
    };

    private static final double[][][] baseIV = {
            {{0, 0, 0}, {1, 1}},
            {{0, 0, 1}, {0, 1}},
            {{0, 1, 0}, {1, 0}},
            {{0, 1, 1}, {0, 1}},
            {{1, 0, 0}, {1, 0}},
            {{1, 0, 1}, {1, 0}},
            {{1, 1, 0}, {1, 0}},
            {{1, 1, 1}, {1, 0}}
    };

    public static void main(String[] argsng) {
        /*
         AND (Base I) - 2 entradas, 1 saída, 2 neurônios ocultos (exemplo)
        */
        MlpImp mlpAnd = new MlpImp(2, 1, 2);
        int total = baseI.length;
        for (int epoch = 0; epoch < 10000; epoch++) {
            double approxErr = 0.0;
            int errCls = 0;
            for (int i = 0; i < baseI.length; i++) {
                double[] out = mlpAnd.train(baseI[i][0], baseI[i][1]);
                approxErr += Math.abs(out[0] - baseI[i][1][0]);
                int pred = out[0] >= 0.5 ? 1 : 0;
                int tgt = (int) baseI[i][1][0];
                if (pred != tgt) errCls++;
            }
            double taxaErroClss = (double) errCls / total;
            System.out.printf("[MLP AND] Epoch: %d - ApproxError: %.4f - ClassError: %d/%d (%.2f%%)%n",
                    epoch, approxErr, errCls, total, taxaErroClss * 100.0);
        }

        /*
         OR (Base II) - 2 entradas, 1 saída
         */
        MlpImp mlpOr = new MlpImp(2, 1, 2);
        total = baseII.length;
        for (int epoch = 0; epoch < 10000; epoch++) {
            double approxErr = 0.0;
            int errCls = 0;
            for (int i = 0; i < baseII.length; i++) {
                double[] out = mlpOr.train(baseII[i][0], baseII[i][1]);
                approxErr += Math.abs(out[0] - baseII[i][1][0]);
                int pred = out[0] >= 0.5 ? 1 : 0;
                int tgt = (int) baseII[i][1][0];
                if (pred != tgt) errCls++;
            }
            double taxaErroClss = (double) errCls / total;
            System.out.printf("[MLP OR] Epoch: %d - ApproxError: %.4f - ClassError: %d/%d (%.2f%%)%n",
                    epoch, approxErr, errCls, total, taxaErroClss * 100.0);
        }

        /*
         XOR (Base III) - 2 entradas, 1 saída (MLP deve resolver)
         */
        MlpImp mlpXor = new MlpImp(2, 1, 3);
        total = baseIII.length;
        for (int epoch = 0; epoch < 10000; epoch++) {
            double approxErr = 0.0;
            int errCls = 0;
            for (double[][] doubles : baseIII) {
                double[] out = mlpXor.train(doubles[0], doubles[1]);
                approxErr += Math.abs(out[0] - doubles[1][0]);
                int pred = out[0] >= 0.5 ? 1 : 0;
                int tgt = (int) doubles[1][0];
                if (pred != tgt) errCls++;
            }
            double taxaErroClss = (double) errCls / total;
            System.out.printf("[MLP XOR] Epoch: %d - ApproxError: %.4f - ClassError: %d/%d (%.2f%%)%n",
                    epoch, approxErr, errCls, total, taxaErroClss * 100.0);
        }

        /*
         Robô (Base IV) - 3 entradas, 2 saídas (avalia acerto completo do vetor)
         */
        MlpImp mlpRobo = new MlpImp(3, 2, 4);
        total = baseIV.length;
        for (int epoch = 0; epoch < 10000; epoch++) {
            double approxErr = 0.0;
            int errCls = 0;
            for (int i = 0; i < baseIV.length; i++) {
                double[] out = mlpRobo.train(baseIV[i][0], baseIV[i][1]);
                boolean ok = true;
                for (int j = 0; j < baseIV[i][1].length; j++) {
                    approxErr += Math.abs(out[j] - baseIV[i][1][j]);
                    int pred = out[j] >= 0.5 ? 1 : 0;
                    int tgt = (int) baseIV[i][1][j];
                    if (pred != tgt) ok = false;
                }
                if (!ok) errCls++;
            }
            double taxaErroClss = (double) errCls / total;
            System.out.printf("[MLP ROBO] Epoch: %d - ApproxError: %.4f - ClassError: %d/%d (%.2f%%)%n",
                    epoch, approxErr, errCls, total, taxaErroClss * 100.0);
        }
    }
}

