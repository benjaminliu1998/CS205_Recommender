# Scala Vs Python Recommender
### Comparing speed, scalability and performance for a distributed movie recommender using ALS

This project aims to compare how effectively Scala and Python implementations of an ALS movie recommender can be accelerated using GPUs with Spark on a cloud-computing platform. We use Spark MLlib to build the Python and Scala recommenders, and we use the NVIDIA spark-rapids package to integrate an AWS EMR cluster with GPUs. We also compare the speedup between a cluster utilizing GPUs and one with only CPUs. Lastly, we compare how well the equivalent Scala and Python implementations perform on the MovieLens datasets of 100k movie ratings, 1M movie ratings, and 20M movie ratings and 25M movie ratings to measure weak scaling.

[See full website](https://benjaminliu1998.github.io/CS205_Recommender/)
