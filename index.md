# Intro

This project aims to compare how effectively scala and python implementations of an ALS movie recommender can be accelerated using GPUs. We use Spark MLlib to build the python and scala recommenders, and we use the NVIDIA spark-rapids package to integrate an AWS EMR cluster with GPUs. We also compare the speed-up between a cluster with GPUs and one with only CPUs. Lastly, we compared how well the equivalent Scala and Python implementations performed on 100k movie ratings, 20M movie ratings and 25M movie ratings to measure weak scaling.

# Problem Definition

## Background to Recommender Systems
Recommender systems address the problem of information overload. (Ullman, 2010) Unlike physical retailers, online outlets have a ‘long-tail’ of esoteric items and therefore cannot show the full range to users because they would be overwhelmed. This has led to the rise of recommender systems, which show only a subset of items to users based on a prediction of what the user wants. The huge number of items (whether it be information, goods or services) that online outlets provide requires large-scale parallel computation so that an appropriate subset of items can be offered to users in a reasonable timeframe.

The rise of recommender systems also brings about the question of algorithmic accountability. The increasing use of neural network recommender systems may mean a greater opacity to how recommendations are made and why they make particular recommendations. Therefore, it is important that systems such as collaborative filtering (which is presented in this project) can provide a viable alternative to neural network recommender systems in the interests of interpretability. It is easier to explore how and why collaborative filtering recommenders make recommendations. Therefore, optimising the efficiency, speed and training of collaborative filtering recommenders is an important task.

## Implicit vs Explicit Recommender Systems
Many recommender systems are based on implicit data. For example, in order to recommend pages for an editor to correct on wikipedia, the recommender might use implicit information about how many times a page has been edited in the past. 

Explicit recommender systems are based on explicit ratings data where a user has deliberately rated items. Although this data is ostensibly more ‘intentional’, the ratings are often very sparse in comparison to implicit data. This means that the recommender will have to predict many separate data points. This further reinforces the importance of a parallel computing infrastructure for this project.


## Content-based Filtering
A technique mainly used in the early days of recommender systems, content-based filtering uses item features to recommend other items similar to what the user likes, based on their previous actions or explicit feedback. While the model can make recommendations solely from a single user’s data, this approach struggles when recommending outside the users’ main interests. It also requires a fair amount of hand-engineering the feature representation of the items to produce a good model, and is typically slower to train than [other approaches.](https://developers.google.com/machine-learning/recommendation/content-based/summary)

## Collaborative Filtering
Collaborative filtering has been a popular method of designing recommender systems, especially since the Netflix Prize in 2009. (Koren, 2009) At the heart of Collaborative Filtering is the idea that the preferences of one user can be modelled based on information from the preferences of other users by calculating the rating similarity between users. 

## ALS
In our application the sparsity and the high dimension of the rating data cause a problem when finding similarity between users. Thus collaborative filtering recommenders rely on matrix factorization to reduce the sparsity and the dimension of the ratings matrix. In this project, we explore the Alternating Least Square (ALS) matrix factorization algorithm. Compared to other matrix factorization algorithms such as Stochastic Gradient Descent, ALS is easier to parallelize, and requires less iterations to reach a good result. The ALS algorithm decomposes the rating matrix A into 2 matrices, W (user factors) and H (movie factors). A user’s rating on a movie will be encoded into latent features in these 2 factored matrices. The idea behind this is that if a user gives good ratings to Avengers, Wonder Woman and Iron Man, these 3 ratings should not be regarded as 3 separate preferences, but a general opinion that the user likes superhero movies. The objective of the ALS algorithm is to minimize the least squares error of the ratings as well as the regularizations (Yu et. al., 2013).

# Big Data and Compute Requirements

The datasets used include up to 25 million movie recommendations, and therefore ALS matrix factorisation can take a long time. As mentioned above, if collaborative filtering is to be a viable (and potentially more interpretable) alternative to neural network-based recommenders, it is important that this application can be scaled to large datasets and perform ALS matrix factorisation quickly. Indeed, as users can come on board a platform very fast, it is critical that any recommender system used at scale would have the capacity to compute user preferences very fast.

A second critical reason that it is important to explore high-end data analytics and big compute solutions for ALS matrix factorisation is that, although matrix factorisation is highly parallelizable, a large body of research suggests that it does not scale well to large datasets (Yu et. al., 2013). ALS can theoretically be easily parallelised iteratively because each of the rows (W and H) can be updated individually. This is outlined above. However, it may not be possible to fit the entirety of row W and H on a node, which limits the potential to parallelise these operations at large scale.

# How our work relates to wider research

There has been much work on parallelising recommender systems, including parallelising ALS. Similarly, there has been work on the differences between Scala and Python for use with Spark. However, according to our literature review, there has been little work done to compare how Scala and Python perform on different GPU configurations, and even less on specifically comparing the parallel performance of Spark and Scala implementations of an ALS movie recommender. Therefore, our project draws on both these bodies of work and aims to add an additional element by exploring the differences between Python and Scala implementations of a parallelised movie recommender using Spark.

## Python versus Scala
There is limited academic work testing the speedup differences between Scala and Python, though within industry it is widely regarded that Scala can achieve a large speedup compared to Python on many tasks. The name ‘Scala’ originally comes from it’s prime function, which is to scale to large quantities of data. Therefore, it’s origins are different to Python, which is a more general-purpose language. Scala is also the native language of Spark, and therefore it may not be surprising that Scala is thought to be faster than Python. Specifically, research points to several reasons that scala Spark programmes can be faster than Pyspark programmes. First, Scala is statically typed, not dynamically typed (like Python) which means the type of variable is known at compile time in Scala programmes. (Ghandi, 2018) This is practical because it means the programmer should be able to catch bugs quicker in Scala than Python, where variables are only known at runtime. It also means that there is additional time spent at runtime understanding the variables in Python, which is not the case with Scala. 

Possibly the most important reason that Scala runs Spark programmes faster than Pyspark is the fact that it uses the JVM. This means that Scala is able to communicate much more effectively with Hadoop, whereas python does not work as well with Hadoop services. There is also evidence to suggest that Scala’s primitives allow it to perform well on programmes involving executing several tasks at the same time. (Ghandi, 2018)

## ALS and Recommender Systems
There has been significant academic work exploring the parallelisation of recommender systems. Indeed, though Stochastic Gradient Descent (SGD) is a faster way to optimise this problem, it is not parallelizable (Yu et. al., 2013). Indeed, researchers have proposed alternatives to both ALS and SGD, such as Fast Parallel Stochastic Gradient Descent (FPSG) and CCD++. Indeed, the CCD++ algorithm was specifically developed to address the fact that ALS does not scale particularly well to large datasets because of the nature of the complexity mentioned above, where there is cubic time complexity for the target rank.

Indeed, one research paper on parallelising ALS matrix factorisation using a GPU compared to CPU found that the use of GPUs resulted in significant overhead, though they observed that the impact of GPUs was greater for larger datasets compared to smaller datasets (Siomos, 2016). 

# Experimental Design and Solution

## Dataset
MovieLens datasets are a series of stable benchmark datasets created by GroupLens Research over the past two decades. (Harper & Konstan, 2015) One of the reasons we wanted to use this dataset is because it is widely used in research on recommender systems and parallelizing Matrix Factorization. (Yu et. al., 2013) Therefore they are easily replicable by other researchers and can be compared to other recommender system results. (For example, see Dooms et. al. 2014; Qiu, 2016; Siomos, 2018)

Some of the new MovieLens datasets include tag genome data with relevance scores. However, we do not use this data for our recommender. The attributes that we use in order to utilize ALS to compute the preferences of users are ‘userId’, ‘rating’ and ‘movieId’. 

## Size
For all datasets we preprocessed the files to remove the header and any columns other than ‘userId’, ‘rating’ and ‘movieId’ because these were not relevant for the recommender; specifically, we removed the ‘timestamp’ column as it caused errors when trying to run the Scala script. Specifically, we used the MovieLens 20M ratings dataset for comparing the GPU and CPU; the dataset includes 27,000 movies by 138,000 users. It was released in 2015 and updated in 2016.

For measuring the recommenders on different dataset sizes we vary the size of the dataset between 100k, 1M, 20M and 25M. They have the following characteristics:

MovieLens 100K  (2.3 MB): 100,000 ratings from 1000 users on 1700 movies
MovieLens 1M dataset (12 MB): 1 million ratings from 6000 users on 4000 movies
MovieLens 20M dataset (305.2 MB): 20 million ratings on 27,000 movies by 138,000 users
MovieLens 25M dataset (390.2 MB): 25 million ratings on 62,000 movies by 162,000 users.

How we distribute ALS Matrix Factorization: MLlib and parallelising ALS matrix factorization for Collaborative Filtering

MLlib is Apache Spark’s machine learning library, which made it an easy decision to use for our solution. MLlib can be easily integrated to use with a cluster and EC2 instances, and it is a Spark Library.

The central scalability problem, as outlined above, is not the parallelisation of ALS as such, but the parallelisation of ALS with large scale data. When the rows that need to iterated over are very large, the task of distributing the data becomes much more difficult. A central reason for using MLlib is to address this challenge through a principled approach. Indeed, there are multiple aspects of the design of MLlib that allow effective implementation of distributed ALS matrix factorisation on large datasets. For example, ‘hybrid partitioning’ is used to decrease the amount of shuffling that happens at each iteration. Critically, ‘block-to-block join’ effectively distributes the user and item matrices across nodes so that you can also decrease the amount of overhead from partitioning (Das et. al., 2016). Specifically, matrices are partitioned into in-blocks and out-blocks in a way such that the parts of each matrice needed for each iteration is accessible.

Both the Scala and Python recommenders use MLlib and are very similar. This was a purposeful decision because we wanted to compare the two recommenders and therefore wanted to make them as similar as possible. Therefore it was important to use the same Spark library (MLlib) for both.

## Accelerating Our Application

The first idea we got for creating a recommender system that utilized GPUs on a Spark cluster came from the CS 205 Spring 2019 project titled Parallelized Amazon Recommendation System Based on [Spark and OpenMP](https://cs205-group11.github.io/amazon-recommendation-system/). In their Suggested Improvements section, they mentioned utilizing GPU instances to try and further facilitate parallelism of their application. After doing some research, we found that while Spark was first created in 2009 and the introduction of AWS with [GPU instances began in November 2010](https://aws.amazon.com/about-aws/whats-new/2010/11/15/announcing-cluster-gpu-instances-for-amazon-ec2/), it hasn’t been until the past few years that people have begun using GPU instances on Spark. One of the big reasons for this increased interest is the rise of data science, and the recognition that data analytics and machine learning can benefit from GPU acceleration to minimize execution time. With the introduction of RAPIDs.AI from NVIDIA that [offered CUDA-integrated software tools in October 2018](https://blogs.nvidia.com/blog/2018/10/10/rapids-data-science-open-source-community/), it became the first platform to offer GPU capabilities for a data science workflow. One branch of the larger RAPIDS open source software libraries and APIs is the RAPIDS Accelerator for Apache Spark, which leverages the power of the RAPIDS [cuDF library](https://github.com/rapidsai/cudf/) and the scale of the Spark distributed computing framework. After looking at the incredible performance and cost benefits posted on the [spark-rapids homepage](https://nvidia.github.io/spark-rapids/) and recognizing that very few people have examined using the combination of Spark and GPUs, we wanted to examine for ourselves if we may be able to replicate some of these performance speedups for our recommender system using ALS. We also wanted to utilize AWS on which we would integrate the GPUs with Spark since it is the most widely used cloud-computing platform, and due to our familiarity with its ecosystem through our continued work on it over the course of this semester.

# How to Replicate our Code

Cluster Details
Release label:emr-6.2.0
Hadoop distribution:Amazon 3.2.1
Applications:Spark 3.0.1, Livy 0.7.0, JupyterEnterpriseGateway 2.1.0
Hardware:
Master: m5.xlarge, 4 vCore, 16 GiB memory, EBS Storage:64 GiB 
Core: g4dn.2xlarge, 8 vCore, 32 GiB memory, 225 SSD GB storage
Task: g4dn.2xlarge, 8 vCore, 32 GiB memory, 225 SSD GB storage

## Setting up a spark-rapids Cluster with GPU

### Software and Configuration
1. Go to AWS EMR.
2. Select ‘create cluster’.
3. Select ‘Advanced Options’.
4. Select emr-6.2.0 for release and ‘Hadoop 3.2.1, Spark 3.0.1, Livy 0.7.0 and JupyterEnterpriseGateway 2.1.0’ for software options.
5. In the “Edit software settings” field enter the following configuration: (Note that spark.task.resource.gpu.amount is set to 1/(number of cores per executor) which allows us to run parallel tasks on the GPU. Therefore, as we dynamically change the number of cores per executor we will also have to change this using the command line.)

[
	{
		"Classification":"spark",
		"Properties":{
			"enableSparkRapids":"true"
		}
	},
	{
		"Classification":"yarn-site",
		"Properties":{
			"yarn.nodemanager.resource-plugins":"yarn.io/gpu",
			"yarn.resource-types":"yarn.io/gpu",
			"yarn.nodemanager.resource-plugins.gpu.allowed-gpu-devices":"auto",
			"yarn.nodemanager.resource-plugins.gpu.path-to-discovery-executables":"/usr/bin",
			"yarn.nodemanager.linux-container-executor.cgroups.mount":"true",
			"yarn.nodemanager.linux-container-executor.cgroups.mount-path":"/sys/fs/cgroup",
			"yarn.nodemanager.linux-container-executor.cgroups.hierarchy":"yarn",
			"yarn.nodemanager.container-executor.class":"org.apache.hadoop.yarn.server.nodemanager.LinuxContainerExecutor"
		}
	},
	{
		"Classification":"container-executor",
		"Properties":{
			
		},
		"Configurations":[
			{
				"Classification":"gpu",
				"Properties":{
					"module.enabled":"true"
				}
			},
			{
				"Classification":"cgroups",
				"Properties":{
					"root":"/sys/fs/cgroup",
					"yarn-hierarchy":"yarn"
				}
			}
		]
	},
	{
        "Classification":"spark-defaults",
        "Properties":{
        "spark.plugins":"com.nvidia.spark.SQLPlugin",
        "spark.sql.sources.useV1SourceList":"",
        "spark.executor.resource.gpu.discoveryScript":"/usr/lib/spark/scripts/gpu/getGpusResources.sh",
        "spark.submit.pyFiles":"/usr/lib/spark/jars/xgboost4j-spark_3.0-1.0.0-0.2.0.jar",
        "spark.executor.extraLibraryPath":"/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/compat/lib:/usr/local/cuda/lib:/usr/local/cuda/lib64:/usr/lib/hadoop/lib/native:/usr/lib/hadoop-lzo/lib/native:/docker/usr/lib/hadoop/lib/native:/docker/usr/lib/hadoop-lzo/lib/native",
        "spark.rapids.sql.concurrentGpuTasks":"2",
        "spark.executor.resource.gpu.amount":"1",
        "spark.executor.cores":"12",
        "spark.task.cpus ":"1",
        "spark.task.resource.gpu.amount":"0.125",
        "spark.rapids.memory.pinnedPool.size":"2G",
        "spark.executor.memoryOverhead":"2G",
        "spark.locality.wait":"0s",
        "spark.sql.shuffle.partitions":"200",
        "spark.sql.files.maxPartitionBytes":"256m",
        "spark.sql.adaptive.enabled":"false"
        }
	},
	{
		"Classification":"capacity-scheduler",
		"Properties":{
			"yarn.scheduler.capacity.resource-calculator":"org.apache.hadoop.yarn.util.resource.DominantResourceCalculator"
		}
	}
]

Cluster Image 1

Cluster Image 2

6. Select the default network and subnet.
7. Change the instance type to g4dn.2xlarge. Select one core and one task instance.

### General Cluster Settings
8. Add a cluster name and an S3 bucket to write cluster logs to.
9. Add a custom ‘Bootstrap Actions’ to allow cgroup permissions to YARN on the cluster. You can use the script at this S3 bucket: s3://recommender-s3-bucket/bootstrap.json

	Alternatively, you could use the script below in your own s3 bucket:

	#!/bin/bash
 
  set -ex
 
  sudo chmod a+rwx -R /sys/fs/cgroup/cpu,cpuacct
  sudo chmod a+rwx -R /sys/fs/cgroup/devices

### Security Settings
10. Select an EC2 key pair.
11. In the “EC2 security groups” tab, confirm that the security group chosen for the “Master” node allows for SSH access. Follow these instructions to allow inbound SSH traffic if the security group does not allow it yet.
12. Select ‘Create Cluster’ and SSH into the Master Node of the Cluster.

[See Spark-Rapids documentation for further details](https://nvidia.github.io/spark-rapids/docs/get-started/getting-started-aws-emr.html)


## Setting up a CPU Cluster
1. Go to AWS EMR.
2. Select ‘create cluster’.
3. Select emr-6.2.0 for release and “Spark” for application, which has Spark 3.0.1, Hadoop 3.2.1.
4. Select m4.2xlarge as instance type, and 3 instances (1 master and 2 core nodes).
5. Choose a cluster name and your key pair. Leave everything else default. Create cluster.
6. Check the summary page is similar to the below image.
7. Go to EC2 -> Instances, find your master node instance, and confirm that the security group chosen for the “Master” node allows for SSH access. Follow these instructions to allow inbound SSH traffic if the security group does not allow it yet.
8. SSH into the Master Node of the Cluster

CPU_Cluster_1 image.

# Scripts
There were two main scripts utilized for this project: recommender.py and recommender.scala. As mentioned previously, the scripts were purposely made to be as similar as possible to best compare execution times. Both scripts contain the variable names and documentation except for where the language syntax differs, and are heavily drawn from the Apache Spark MLlib examples repository (Scala: https://github.com/apache/spark/blob/master/examples/src/main/scala/org/apache/spark/examples/mllib/RecommendationExample.scala Python: https://github.com/apache/spark/blob/master/examples/src/main/python/mllib/recommendation_example.py). The high-level overview of the script is as follows: create a SparkContext, read in the .csv file, map the dataset to an RDD in the form required for the ALS() function. train the ALS on the RDD, make predictions based on the user-movie tuple, and compare the true user-movie ratings with the predicted user-movie ratings from the ALS using mean squared error. We recognize that a more robust ALS prediction model can be made which contains a train-test split, but our focus for this project was execution time comparisons; therefore, we were content as long as each script produced similar mean squared errors depending on the dataset used. 
Two other scripts created for this project were the build.sbt used for the GPU, and the build.sbt used for the CPU. The extension .sbt stands for Simple Build Tool, and it is an open-source build tool for Scala and Java projects that allows for easily compiling and creating .jar files for projects. The build.sbt file contains metadata information about the project, as well as all dependencies that are required to run the code. We had to create two different versions due to different scala versions on the two different clusters (GPU had Scala version 2.12.10 available, while CPU had Scala version 2.11.12 available). We don’t expect there to be any differences in execution time as a result of these two different library versions for the two different clusters since no updates have been made to the ALS functions used between the two version times.
