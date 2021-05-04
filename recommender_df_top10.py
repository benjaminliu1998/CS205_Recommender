# Inspired by https://github.com/shashwatwork/Building-Recommeder-System-in-PySpark/blob/master/Crafting%20Recommedation%20System%20with%20PySpark.ipynb

from pyspark.mllib.recommendation import MatrixFactorizationModel, Rating
from pyspark.ml.recommendation import ALS
from pyspark import SparkConf, SparkContext
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('rec').getOrCreate()

df=spark.read.csv('ratings_head.csv',inferSchema=True, header=True)

(train, val) = df.randomSplit([0.8, 0.2])

# Build the recommendation model using Alternating Least Squares
als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating",
          coldStartStrategy="drop")
model = als.fit(train)

predictions = model.transform(val)

evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Model Root-mean-square error = " + str(rmse))


# Need to ensure that this userid is in the validation set.
single_user = val.filter(val['userId']==1260759144).select(['movieId','userId'])
single_user_2 = train.filter(train['userId']==1260759144).select(['movieId','userId'])

reccomendations = model.transform(single_user)
reccomendations_2 = model.transform(single_user_2)

reccomendations.orderBy('prediction',ascending=False).show()
reccomendations_2.orderBy('prediction',ascending=False).show()
