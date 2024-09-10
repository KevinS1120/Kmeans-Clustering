/*
    Kevin Stinnett
    Data Clustering

    Java best practices: https://blog.jetbrains.com/idea/2024/02/java-best-practices/
 */
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.*;

public class phase5 {
    //I declared numClusters up here so it can be used throughout the program easier since it will be taken from the text files now
    private static int numClusters;
    //The fileReader class will read the file and store the data points in a double array to return for use in other classes.
    private static List<double[]> fileReader(String fileName) {
        List<double[]> data = new ArrayList<>();    //Here I created a double array to take in the data from each line of the file.

        //Created a scanner called fileScanner which will be going through the file until it reaches the end
        try (Scanner fileScanner = new Scanner(new File(fileName), StandardCharsets.UTF_8)) {
            String firstLine = fileScanner.nextLine().trim();
            Scanner firstLineScanner = new Scanner(firstLine);  //Created another scanner solely to knock out the first line and save the values of numPoints and dimensionality
            int dimensionality = firstLineScanner.nextInt();
            firstLineScanner.nextInt();
            //Take in the numClusters from the first line of the text file
            numClusters = firstLineScanner.nextInt();
            firstLineScanner.close();

            //While there are still lines in the file, the fileScanner will add it to the fileLines ArrayList.
            while (fileScanner.hasNext()) {
                String fileLine = fileScanner.nextLine();
                String[] individualPoints = fileLine.split(" ");        //I used this delimiter to split the line into each data point everywhere there is a space because that's what's between each point.

                //Because dimensionality is the number of data points on each line of the file, this loop runs until the line ends.
                double[] dataPoints = new double[dimensionality];       //Stores the data values
                int i = 0;
                for (String point : individualPoints) {
                    dataPoints[i++] = Double.parseDouble(point);        //This line takes each of the data points that were separated through my delimiter and adds it to a double array for storage
                }
                data.add(dataPoints);       //Add those data points to the ArrayList I declared earlier
            }
            //Created this catch statement in case there are any exceptions such as trouble reading the file
        } catch (IOException e) {
            System.exit(1);
        }
        return data;
    }

    //Performs the normalization of the data
    private static double[][] minMaxNormalization(List<double[]> data) {
        //Initialize variables numRows and numColumns for use
        int numRows = data.size();
        int numColumns = data.get(0).length;
        //normalizedData array of doubles to hold the values
        double[][] normalizedData = new double[numRows][numColumns];
        //For formula
        double newMin = 0.0;
        double newMax = 1.0;

        //Performs the min-max normalization formula on the data set
        for (int i = 0; i < numColumns; i++) {
            double minData = Double.MAX_VALUE;
            double maxData = Double.MIN_VALUE;

            //For loop to find min and max values for each column
            for (int j = 0; j < numRows; j++) {
                if (data.get(j)[i] < minData) minData = data.get(j)[i];
                if (data.get(j)[i] > maxData) maxData = data.get(j)[i];
            }

            //Min-max normalization formula
            for (int j = 0; j < numRows; j++) {
                if (maxData == minData) {
                    normalizedData[j][i] = newMin; //For avoiding division by zero
                } else {
                    //Otherwise, just perform the formula on the data
                    normalizedData[j][i] = (data.get(j)[i] - minData) / (maxData - minData) * (newMax - newMin) + newMin;
                }
            }
        }
        //Return the array holding the now normalized data for use in calculations
        return normalizedData;
    }

    //This clusterSelector class is responsible for the unique random selection of the clusters, it then returns them to be referenced by other classes.
    private static Set<Integer> clusterSelector(List<double[]> data, int numClusters){
        Random randomPicker = new Random();     //Here I initialized the randomPicker which will randomly select the clusters needed
        HashSet<Integer> clusters = new HashSet<>();        //I decided to use a HashSet to store the clusters in because they don't allow duplicate values

        //This while loop will run until the number of clusters passed into the program by the user is met
        while (clusters.size() < numClusters) {
            int selectedClusterLocation = randomPicker.nextInt(data.size());        //Randomly selects the cluster from the available data in the file
            clusters.add(selectedClusterLocation);      //Adds the selected cluster to the HashSet
        }
        return clusters;
    }

    //For performing random selection on the data set
    private static double[][] randomSelectionInitialization(List<double[]> data, int k) {
        //Get number of dimensions for each data point
        int numDimensions = data.get(0).length;
        //Initialize array to hold "k" centroids
        double[][] centers = new double[k][numDimensions];
        //Use clusterSelector to randomly select the unique cluster locations
        Set<Integer> selectedLocations = clusterSelector(data, k);

        //Counter
        int i = 0;
        //Iterates through the selected centroid locations and sets the data point at that location to be the centroid
        for (int index : selectedLocations) {
            centers[i++] = data.get(index);
        }
        //Return the centroids
        return centers;
    }

    //Function that calculates the squared Euclidean distance which is used to determine the closest centroid that a point can be assigned to
    //Also used in calculating the SSE
    private static double squaredEuclideanDistance(double[] point1, double[] point2) {
        double distance = 0.0;

        for(int i = 0; i < point1.length; i++) {
            distance += (point1[i] - point2[i]) * (point1[i] - point2[i]);
        }
        return distance;
    }

    //The SSE is the sum of the distances between all of the points for a specific cluster (distance between each point to the cluster)
    private static double calculateSSE(List<double[]> data, int[] clusterAssignment, double[][] centroids) {
        double sse = 0.0;

        //Calculated the squared Euclidean distance between the current data point (i) and the nearest centroid
        for(int i = 0; i < data.size(); i++) {
            sse += squaredEuclideanDistance(data.get(i), centroids[clusterAssignment[i]]);
        }
        return sse;
    }

    //Calculates the Rand External Validity Index using the formula R = (a+b)/(n/2)
    private static double calculateRandIndex(int[] trueLabelValues, int[] predictedLabelValues) {
        int n = trueLabelValues.length;
        int a = 0; //a in this formula is the count of element pairs belonging to the SAME cluster
        int b = 0; //b in this formula is the count of element pairs belonging to DIFFERENT clusters

        //Iterates through all of the points
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                //trueLabelValues is for checking the "true" cluster assignments, the true clusters are from the dataset files
                boolean trueClustering = trueLabelValues[i] == trueLabelValues[j];
                //predictedLabelValues is for checking the accuracy of the algorithm and for comparing it to the true clustering from the dataset
                boolean predictedClustering = predictedLabelValues[i] == predictedLabelValues[j];

                //If the points are both in the true cluster AND predicted cluster from the program, increment a
                if (trueClustering && predictedClustering) {
                    a++;
                    //Otherwise, it will increment b where the points are in different clusters
                } else if (!trueClustering&& !predictedClustering) {
                    b++;
                }
            }
        }

        //totalOptions is the total number of possible unique pairs of points
        int totalOptions = (n * (n - 1)) / 2;
        //Calculate and return the actual value of the Rand index
        return (double) (a + b) / totalOptions;
    }

    //This function calculates the Jaccard external validation index using the formula
    private static double calculateJaccardIndex(int[] trueLabels, int[] predictedLabels) {
        int n = trueLabels.length;
        int TP = 0; //True Positives
        int FP = 0; //False Positives
        int FN = 0; //False Negatives

        //Checks for the true positives, false positives, and false negatives
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                boolean sameTrueCluster = trueLabels[i] == trueLabels[j];
                boolean samePredictedCluster = predictedLabels[i] == predictedLabels[j];

                if (sameTrueCluster && samePredictedCluster) {
                    TP++;
                } else if (!sameTrueCluster && samePredictedCluster) {
                    FP++;
                } else if (sameTrueCluster && !samePredictedCluster) {
                    FN++;
                }
            }
        }

        //Compute the Jaccard external validation Index
        double jaccardIndex = TP + FP + FN;
        return jaccardIndex = TP / jaccardIndex;
    }

    private static double[] runAlgorithm(List<double[]> data, int numClusters, int maxIterations, double convergenceThreshold, double[][] initialCentroids) {
        int numPoints = data.size();    //Number of data points in the file
        int dimensionality = data.get(0).length;    //The dimensionality is the first number in each file
        double[][] centroids = initialCentroids;    //Create an array of doubles to store the centroids
        int[] clusterAssignments = new int[numPoints];  //clusterAssignments array holds the location of the centroid where the data points are assigned
        double finalSSE = Double.MAX_VALUE;  //Initialized the variable for the previous SSE, which is for now just set to a high number for comparison reasons

        for (int iteration = 0; iteration < maxIterations; iteration++) {
            //Assign points to clusters
            for (int i = 0; i < numPoints; i++) {
                double minDistance = Double.MAX_VALUE;  //Initialize a minimum distance variable and for now just set it to a high number to be compared to the actual calculated distances
                int bestCluster = 0;        //bestCluster will be the cluster closest to the given data point

                //For each cluster, computes the squared Euclidean Distance between it and the centroid
                for (int j = 0; j < numClusters; j++) {
                    //If the computed distance is less than the minimum distance value from earlier, it becomes the new minimum distance and this cluster will be marked as the closest centroid
                    double dist = squaredEuclideanDistance(data.get(i), centroids[j]);
                    if (dist < minDistance) {
                        minDistance = dist;
                        bestCluster = j;
                    }
                }
                clusterAssignments[i] = bestCluster;    //Updated to hold the location of the closest centroid
            }

            //Update the centroids
            double[][] newCentroids = new double[numClusters][dimensionality];
            int[] pointsPerCluster = new int[numClusters];

            for (int i = 0; i < numPoints; i++) {
                int cluster = clusterAssignments[i];
                pointsPerCluster[cluster]++;
                for (int j = 0; j < dimensionality; j++) {
                    newCentroids[cluster][j] += data.get(i)[j];
                }
            }

            //Goes through all of the data points
            for (int i = 0; i < numPoints; i++) {
                int cluster = clusterAssignments[i];       //cluster holds the location of the centroid that the current data point is assigned to
                pointsPerCluster[cluster]++;     //Keep track of the number of points assigned to the specific cluster to find the average later
                //Here the coordinates of the current data point are being added to get the total coordinates for average purposes
                for (int j = 0; j < dimensionality; j++) {
                    newCentroids[cluster][j] += data.get(i)[j];
                }
            }

            //Calculate the sse of the new centroids
            centroids = newCentroids;
            double currentSSE = calculateSSE(data, clusterAssignments, centroids);

            //Checks for when to stop the program
            if ((finalSSE - currentSSE) / finalSSE < convergenceThreshold) {
                double[] result = new double[2 + clusterAssignments.length];
                result[0] = currentSSE;
                result[1] = iteration + 1;
                //Store the results
                for (int i = 0; i < clusterAssignments.length; i++) {
                    result[i + 2] = clusterAssignments[i];
                }
                return result;
            }
            finalSSE = currentSSE;
        }
        double[] result = new double[2 + clusterAssignments.length];
        result[0] = finalSSE;
        result[1] = maxIterations;
        for (int i = 0; i < clusterAssignments.length; i++) {
            result[i + 2] = clusterAssignments[i];
        }

        //Return output
        return new double[]{finalSSE, maxIterations};
    }

    //finalOutput function is for running the algorithm with the correct input parameters, it also performs the two external validation index functions.
    private static void finalOutput(String filename, List<double[]> data, int[] trueLabels, int numClusters, int maxIterations, double convergenceThreshold, int numRuns) {
        //Normalize the data
        double[][] normalizedDataArray = minMaxNormalization(data);
        List<double[]> normalizedData = new ArrayList<>();  //ArrayList that holds the normalized data
        for (double[] row : normalizedDataArray) {
            normalizedData.add(row);
        }

        System.out.println("\n\nCalculations for file: " + filename);
        System.out.println("----------------------------------------------------");

        //Initialize variables for tracking purposes
        double rsBestInitialSSE = Double.MAX_VALUE;
        double rsBestFinalSSE = Double.MAX_VALUE;
        int rsBestIterations = 0;
        double bestRandIndex = 0.0;
        double bestJaccardIndex = 0.0;

        //Performs k-means clustering using random selection
        for (int run = 1; run <= numRuns; run++) {
            //Randomly select initial centroids
            double[][] initialCentroids = randomSelectionInitialization(normalizedData, numClusters);
            //Calculate the initial SSE
            double initialSSE = calculateSSE(normalizedData, new int[normalizedData.size()], initialCentroids);
            //Perform the algorithm, and store the best results
            double[] result = runAlgorithm(normalizedData, numClusters, maxIterations, convergenceThreshold, initialCentroids);
            double finalSSE = result[0];
            int numIterations = (int) result[1];

            //Takes the cluster assignments, has to increment by 2 because of points
            int[] clusterAssignments = new int[normalizedData.size()];
            for (int i = 0; i < clusterAssignments.length; i++) {
                clusterAssignments[i] = (int) result[i + 2];
            }

            //Calculate the rand external validation index
            double randIndex = calculateRandIndex(trueLabels, clusterAssignments);
            //Calculate the jaccard external validation index
            double jaccardIndex = calculateJaccardIndex(trueLabels, clusterAssignments);

            //Comparisons
            if (finalSSE < rsBestFinalSSE) {
                rsBestInitialSSE = initialSSE;
                rsBestFinalSSE = finalSSE;
                rsBestIterations = numIterations;
                bestRandIndex = randIndex;
                bestJaccardIndex = jaccardIndex;
            }
        }
        //Output the values for the two external validation indices
        System.out.println("Best Rand Index: " + bestRandIndex);
        System.out.println("Best Jaccard Index: " + bestJaccardIndex);
    }

    //Main
    public static void main(String[] args) {
        //Only takes in the file name
        if (args.length != 1) {
            System.out.println("Incorrect format, only include <F>");
            System.exit(1);
        }

        //Declare the file name as the user input
        String fileName = args[0];
        //Store these values because they're the same for every run
        int maxIterations = 100;
        double convergenceThreshold = 0.001;
        int numRuns = 100;

        //Reads in the data
        List<double[]> data = fileReader(fileName);

        //Stores the true clusters
        int[] trueLabels = new int[data.size()];
        for (int i = 0; i < trueLabels.length; i++) {
            trueLabels[i] = i % numClusters;
        }

        //Run the algorithm and output results
        finalOutput(fileName, data, trueLabels, numClusters, maxIterations, convergenceThreshold, numRuns);
    }
}