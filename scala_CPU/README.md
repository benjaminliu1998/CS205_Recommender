## Scala Recommender Script for GPU Cluster

The high-level overview of the script is as follows: create a SparkContext, read in the .csv file, map the dataset to an RDD in the form required for the ALS() function. train the ALS on the RDD, make predictions based on the user-movie tuple, and compare the true user-movie ratings with the predicted user-movie ratings from the ALS using mean squared error.

