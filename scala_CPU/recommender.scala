/**
 * this code is based on the example provided on the Apach Spark website:
 * https://github.com/apache/spark/blob/master/examples/src/main/scala/org/apache/spark/examples/mllib/RecommendationExample.scala
 */

// importing the necessary packages
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating

object RecommenderScala {
  /**
   * This program receives a MovieLens dataset, changes it into a RDD file to be entered in the
   * ALS function, and then fits and evaluates the performance of the given model 
   *
   */
  def main(args: Array[String]): Unit = {
    /**
     * applications should define a main() method instead of extending scala.App since subclasses
     * of scala.App may not work properly in this instance
     */

    // set the class which gives the various options to provide configuration parameters
    val conf = new SparkConf().setAppName("RecommenderScala")
    // set the entry point for the Spark environment
    val sc = new SparkContext(conf)

    // Load the dataset
    val data = sc.textFile(args(0))

    // map the dataset into an RDD file where the users and movies are integer types, and the ratings is a double type
    val ratings = data.map(_.split(',') match { case Array(user, movie, rating) =>
      Rating(user.toInt, movie.toInt, rating.toDouble)
    })

    // Build the recommender system using ALS: see https://spark.apache.org/docs/latest/ml-collaborative-filtering.html for more hyperparameter tuning options
    val rank = 8 // number of latent factors in the model (defaults to 10)
    val numIterations = 15 // Number of iterations of ALS.(defaults to 5)

    // training the ALS model
    val model = ALS.train(ratings, rank, numIterations)

    // creating a user-movie tuple from the ratings dataset to prepare for evaluation
    val usersMovies = ratings.map { case Rating(user, movie, rating) =>
      (user, movie)
    }
    // predicting the rating for a given user and movie
    val predictions =
      model.predict(usersMovies).map { case Rating(user, movie, rating) =>
        ((user, movie), rating)
      }
    // joining the predictions dataset with the the ratings dataset to compare how well the true rating compared to the predicted rating
    val ratesAndPreds = ratings.map { case Rating(user, movie, rating) =>
      ((user, movie), rating)
    }.join(predictions)
    // evaluating the performance of the true rating compared to the predicted rating using MSE
    val MSE = ratesAndPreds.map { case ((user, movie), (r1, r2)) =>
      val err = (r1 - r2)
      err * err
    }.mean()
    println(s"Mean Squared Error = $MSE")

    // manually stopping the SparkContext to let the application know it's done consuming resources (can also set sc = SparkContext.getOrCreate(conf) in line 27 to avoid doing this manually)
    sc.stop()
  }
}
