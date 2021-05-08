'''
this code is based on the example provided on the Apach Spark website: https://github.com/apache/spark/blob/master/examples/src/main/python/mllib/recommendation_example.py
'''


# importing the necessary packages
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from pyspark import SparkConf, SparkContext
import sys

# set the class which gives the various options to provide configuration parameters
conf = SparkConf().setAppName('RecommenderPython')
# set the entry point for the Spark environment
sc = SparkContext(conf = conf)

# Load the dataset
data = sc.textFile(sys.argv[1])

# map the dataset into an RDD file where the users and movies are integer types, and the ratings is a double type
ratings = data.map(lambda l: l.split(','))\
    .map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))

# Build the recommender system using ALS: see https://spark.apache.org/docs/latest/ml-collaborative-filtering.html for more hyperparameter tuning options
rank = 8 # number of latent factors in the model (defaults to 10)
numIterations = 15 # Number of iterations of ALS.(defaults to 5)

# training the ALS model
model = ALS.train(ratings, rank, numIterations)

# creating a user-movie tuple from the ratings dataset to prepare for evaluation
usersMovies = ratings.map(lambda p: (p[0], p[1]))
# predicting the rating for a given user and movie
predictions = model.predict(usersMovies).map(lambda r: ((r[0], r[1]), r[2]))
# joining the predictions dataset with the the ratings dataset to compare how well the true rating compared to the predicted rating
ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
# evaluating the performance of the true rating compared to the predicted rating using MSE
MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
print("Mean Squared Error = " + str(MSE))

# manually stopping the SparkContext to let the application know it's done consuming resources (can also set sc = SparkContext.getOrCreate(conf) in line 27 to avoid doing this manually)
sc.stop()
