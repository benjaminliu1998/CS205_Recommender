name := "Scala Recommender"

// version of the application
version := "1.0"

// version of Scala that is being used
scalaVersion := "2.12.10"

// libraries that are needed to run ALS
libraryDependencies += "org.apache.spark" %% "spark-sql" % "3.0.1"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "3.0.1"
libraryDependencies += "org.apache.spark" %% "spark-core" % "3.0.1"
