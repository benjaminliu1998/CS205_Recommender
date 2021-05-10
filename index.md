Intro

This project aims to compare how effectively scala and python implementations of an ALS movie recommender can be accelerated using GPUs. We use Spark MLlib to build the python and scala recommenders, and we use the NVIDIA spark-rapids package to integrate an AWS EMR cluster with GPUs. We also compare the speed-up between a cluster with GPUs and one with only CPUs. Lastly, we compared how well the equivalent Scala and Python implementations performed on 100k movie ratings, 20M movie ratings and 25M movie ratings to measure weak scaling.

Problem Definition

Background to Recommender Systems
Recommender systems address the problem of information overload. (Ullman, 2010) Unlike physical retailers, online outlets have a ‘long-tail’ of esoteric items and therefore cannot show the full range to users because they would be overwhelmed. This has led to the rise of recommender systems, which show only a subset of items to users based on a prediction of what the user wants. The huge number of items (whether it be information, goods or services) that online outlets provide requires large-scale parallel computation so that an appropriate subset of items can be offered to users in a reasonable timeframe.

The rise of recommender systems also brings about the question of algorithmic accountability. The increasing use of neural network recommender systems may mean a greater opacity to how recommendations are made and why they make particular recommendations. Therefore, it is important that systems such as collaborative filtering (which is presented in this project) can provide a viable alternative to neural network recommender systems in the interests of interpretability. It is easier to explore how and why collaborative filtering recommenders make recommendations. Therefore, optimising the efficiency, speed and training of collaborative filtering recommenders is an important task.

Implicit vs Explicit Recommender Systems
Many recommender systems are based on implicit data. For example, in order to recommend pages for an editor to correct on wikipedia, the recommender might use implicit information about how many times a page has been edited in the past. 

Explicit recommender systems are based on explicit ratings data where a user has deliberately rated items. Although this data is ostensibly more ‘intentional’, the ratings are often very sparse in comparison to implicit data. This means that the recommender will have to predict many separate data points. This further reinforces the importance of a parallel computing infrastructure for this project.
