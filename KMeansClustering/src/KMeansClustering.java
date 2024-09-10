import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

public class KMeansClustering {

    public static void main(String[] args) {
        // Example data sets
        String[] dataSets = {"DataSet1", "DataSet2", "DataSet3", "DataSet4", "DataSet5", "DataSet6", "DataSet7", "DataSet8", "DataSet9", "DataSet10"};

        // Normalization method
        String normalizationMethod = "Min-Max"; // or "Z-Score"

        // Initialization methods
        String[] initMethods = {"Random Selection", "Random Partition"};

        // Performance measurements
        double[][] initialSSE = new double[dataSets.length][initMethods.length];
        double[][] finalSSE = new double[dataSets.length][initMethods.length];
        int[][] iterations = new int[dataSets.length][initMethods.length];

        // Parameters for K-Means
        int k = 3; // Number of clusters
        int maxIterations = 100;
        double convergenceThreshold = 0.01;

        // Loop through each data set
        for (int i = 0; i < dataSets.length; i++) {
            double[][] data = loadData(dataSets[i]);
            double[][] normalizedData = normalizeData(data, normalizationMethod);

            // Loop through each initialization method
            for (int j = 0; j < initMethods.length; j++) {
                double[][] initialCentroids;
                if (initMethods[j].equals("Random Selection")) {
                    initialCentroids = randomSelectionInitialization(normalizedData, k);
                } else {
                    initialCentroids = randomPartitionInitialization(normalizedData, k);
                }

                KMeansResult result = runKMeans(normalizedData, initialCentroids, maxIterations, convergenceThreshold);
                initialSSE[i][j] = result.initialSSE;
                finalSSE[i][j] = result.finalSSE;
                iterations[i][j] = result.iterations;
            }
        }

        // Write results to CSV file
        try (FileWriter csvWriter = new FileWriter("kmeans_results.csv")) {
            csvWriter.append("Data Set,Normalization Method,Initialization Method,Initial SSE,Final SSE,Number of Iterations\n");
            for (int i = 0; i < dataSets.length; i++) {
                for (int j = 0; j < initMethods.length; j++) {
                    csvWriter.append(dataSets[i])
                            .append(",")
                            .append(normalizationMethod)
                            .append(",")
                            .append(initMethods[j])
                            .append(",")
                            .append(String.valueOf(initialSSE[i][j]))
                            .append(",")
                            .append(String.valueOf(finalSSE[i][j]))
                            .append(",")
                            .append(String.valueOf(iterations[i][j]))
                            .append("\n");
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // Load data (placeholder implementation)
    private static double[][] loadData(String dataSet) {
        // Example data; replace with actual data loading logic
        return new double[][]{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}, {7.0, 8.0}, {9.0, 10.0}};
    }

    // Normalize data based on method
    private static double[][] normalizeData(double[][] data, String method) {
        if (method.equals("Min-Max")) {
            return minMaxNormalization(data);
        } else if (method.equals("Z-Score")) {
            return zScoreNormalization(data);
        } else {
            throw new IllegalArgumentException("Unknown normalization method: " + method);
        }
    }

    // Min-Max normalization
    private static double[][] minMaxNormalization(double[][] data) {
        int rows = data.length;
        int cols = data[0].length;
        double[][] normalizedData = new double[rows][cols];

        for (int j = 0; j < cols; j++) {
            double min = Double.MAX_VALUE;
            double max = Double.MIN_VALUE;

            for (int i = 0; i < rows; i++) {
                if (data[i][j] < min) min = data[i][j];
                if (data[i][j] > max) max = data[i][j];
            }

            double range = max - min;
            for (int i = 0; i < rows; i++) {
                normalizedData[i][j] = (range == 0) ? 0 : (data[i][j] - min) / range;
            }
        }
        return normalizedData;
    }

    // Z-Score normalization
    private static double[][] zScoreNormalization(double[][] data) {
        int rows = data.length;
        int cols = data[0].length;
        double[][] normalizedData = new double[rows][cols];

        for (int j = 0; j < cols; j++) {
            double mean = 0;
            double stdDev = 0;

            for (int i = 0; i < rows; i++) {
                mean += data[i][j];
            }
            mean /= rows;

            for (int i = 0; i < rows; i++) {
                stdDev += Math.pow(data[i][j] - mean, 2);
            }
            stdDev = Math.sqrt(stdDev / rows);

            for (int i = 0; i < rows; i++) {
                normalizedData[i][j] = (stdDev == 0) ? 0 : (data[i][j] - mean) / stdDev;
            }
        }
        return normalizedData;
    }

    // Random Selection initialization
    private static double[][] randomSelectionInitialization(double[][] data, int k) {
        Random rand = new Random();
        double[][] centroids = new double[k][data[0].length];
        for (int i = 0; i < k; i++) {
            centroids[i] = data[rand.nextInt(data.length)];
        }
        return centroids;
    }

    // Random Partition initialization
    private static double[][] randomPartitionInitialization(double[][] data, int k) {
        Random rand = new Random();
        double[][] centroids = new double[k][data[0].length];
        int[] clusterSizes = new int[k];
        double[][] sums = new double[k][data[0].length];

        for (int i = 0; i < data.length; i++) {
            int cluster = rand.nextInt(k);
            clusterSizes[cluster]++;
            for (int j = 0; j < data[i].length; j++) {
                sums[cluster][j] += data[i][j];
            }
        }

        for (int i = 0; i < k; i++) {
            for (int j = 0; j < data[0].length; j++) {
                centroids[i][j] = (clusterSizes[i] == 0) ? 0 : sums[i][j] / clusterSizes[i];
            }
        }
        return centroids;
    }

    // Run K-Means algorithm
    private static KMeansResult runKMeans(double[][] data, double[][] initialCentroids, int maxIterations, double convergenceThreshold) {
        int n = data.length;
        int k = initialCentroids.length;
        int dim = data[0].length;
        double[][] centroids = new double[k][dim];
        int[] assignments = new int[n];
        double initialSSE = 0.0;
        double finalSSE = 0.0;

        // Copy initial centroids
        for (int i = 0; i < k; i++) {
            System.arraycopy(initialCentroids[i], 0, centroids[i], 0, dim);
        }

        for (int iter = 0; iter < maxIterations; iter++) {
            // Assignment step
            boolean changed = false;
            initialSSE = 0.0;
            for (int i = 0; i < n; i++) {
                double minDist = Double.MAX_VALUE;
                int closestCluster = -1;
                for (int j = 0; j < k; j++) {
                    double dist = 0.0;
                    for (int d = 0; d < dim; d++) {
                        dist += Math.pow(data[i][d] - centroids[j][d], 2);
                    }
                    if (dist < minDist) {
                        minDist = dist;
                        closestCluster = j;
                    }
                }
                if (assignments[i] != closestCluster) {
                    changed = true;
                    assignments[i] = closestCluster;
                }
                initialSSE += minDist;
            }

            if (!changed) {
                break;
            }

            // Update step
            double[][] newCentroids = new double[k][dim];
            int[] counts = new int[k];

            for (int i = 0; i < n; i++) {
                int cluster = assignments[i];
                counts[cluster]++;
                for (int d = 0; d < dim; d++) {
                    newCentroids[cluster][d] += data[i][d];
                }
            }

            for (int j = 0; j < k; j++) {
                for (int d = 0; d < dim; d++) {
                    if (counts[j] != 0) {
                        newCentroids[j][d] /= counts[j];
                    } else {
                        newCentroids[j][d] = centroids[j][d]; // Keep old centroid if no points assigned
                    }
                }
            }

            // Calculate final SSE
            finalSSE = 0.0;
            for (int i = 0; i < n; i++) {
                double dist = 0.0;
                for (int d = 0; d < dim; d++) {
                    dist += Math.pow(data[i][d] - newCentroids[assignments[i]][d], 2);
                }
                finalSSE += dist;
            }

            // Check convergence
            boolean converged = true;
            for (int j = 0; j < k; j++) {
                for (int d = 0; d < dim; d++) {
                    if (Math.abs(newCentroids[j][d] - centroids[j][d]) > convergenceThreshold) {
                        converged = false;
                        break;
                    }
                }
                if (!converged) {
                    break;
                }
            }
            if (converged) {
                break;
            }

            centroids = newCentroids;
        }

        return new KMeansResult(initialSSE, finalSSE, maxIterations);
    }

    // Placeholder class for K-Means results
    private static class KMeansResult {
        double initialSSE;
        double finalSSE;
        int iterations;

        KMeansResult(double initialSSE, double finalSSE, int iterations) {
            this.initialSSE = initialSSE;
            this.finalSSE = finalSSE;
            this.iterations = iterations;
        }
    }
}
