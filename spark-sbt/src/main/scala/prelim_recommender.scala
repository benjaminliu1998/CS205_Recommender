import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.sql.SparkSession


object Simple {

	def main(args: Array[String]) {

		// Load and parse the data
		val spark = SparkSession.builder.appName("Simple Application").getOrCreate()
		import spark.implicits._
		val data = spark.read.textFile("test_2.data")
		val ratings = data.map(_.split(',') match { case Array(user, item, rate) =>
		  Rating(user.toInt, item.toInt, rate.toDouble)
		})

		// Build the recommendation model using ALS
		val rank = 8
		val numIterations = 15
		val model = ALS.train(ratings, rank, numIterations)

		// Evaluate the model on rating data
		val usersProducts = ratings.map { case Rating(user, product, rate) =>
		  (user, product)
		}
		val predictions =
		  model.predict(usersProducts).map { case Rating(user, product, rate) =>
		    ((user, product), rate)
		  }
		val ratesAndPreds = ratings.map { case Rating(user, product, rate) =>
		  ((user, product), rate)
		}.join(predictions)
		val MSE = ratesAndPreds.map { case ((user, product), (r1, r2)) =>
		  val err = (r1 - r2)
		  err * err
		}.mean()
		println(s"Mean Squared Error = $MSE")

		spark.stop()
	}

}