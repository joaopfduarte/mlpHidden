package converter;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class VertebralColumnConverter {
    private final String filePathTrain = "src/main/java/base/vertebral+column/train.csv";
    private final String filePathTest = "src/main/java/base/vertebral+column/test.csv";

    public List<Register> trainConverter() {
        List<Register> dataset = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(filePathTrain))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.isBlank()) continue;
                String[] tokens = splitLine(line);
                if (tokens.length < 2) continue;

                String classToken = tokens[tokens.length - 1];

                double[] features = new double[tokens.length - 1];
                for (int i = 0; i < tokens.length - 1; i++) {
                    features[i] = Double.parseDouble(tokens[i]);
                }

                double[] target = new double[] { "AB".equalsIgnoreCase(classToken) ? 1.0 : 0.0 };

                dataset.add(new Register(features, target));
            }
        } catch (IOException e) {
            System.err.println("Error reading file: " + e.getMessage());
        }
        return dataset;
    }

    public List<Register> testConverter() {
        List<Register> dataset = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(filePathTest))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.isBlank()) continue;
                String[] tokens = splitLine(line);
                if (tokens.length < 2) continue;

                String classToken = tokens[tokens.length - 1];

                double[] features = new double[tokens.length - 1];
                for (int i = 0; i < tokens.length - 1; i++) {
                    features[i] = Double.parseDouble(tokens[i]);
                }

                double[] target = new double[] { "AB".equalsIgnoreCase(classToken) ? 1.0 : 0.0 };

                dataset.add(new Register(features, target));
            }
        } catch (IOException e) {
            System.err.println("Error reading file: " + e.getMessage());
        }
        return dataset;
    }

    private String[] splitLine(String line) {
        String[] parts = line.trim().split(",");
        for (int i = 0; i < parts.length; i++) {
            parts[i] = parts[i].trim();
        }
        return parts;
    }
}
