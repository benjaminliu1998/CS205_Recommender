## Scala Recommender Script for CPU Cluster

The high-level overview of the script is as follows: create a SparkContext, read in the .csv file, map the dataset to an RDD in the form required for the ALS() function, train the ALS on the RDD, make predictions based on the user-movie tuple, and compare the true user-movie ratings with the predicted user-movie ratings from the ALS using mean squared error.

The extension .sbt stands for Simple Build Tool, and it is an open-source build tool for Scala and Java projects that allows for easily compiling and creating .jar files for projects. The build.sbt file contains metadata information about the project, as well as all dependencies that are required to run the code. The version used for the CPU cluster is based on Scala version 2.12.10.
