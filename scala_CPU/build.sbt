name := "Scala Recommender"

// version of the application
version := "1.0"

// version of Scala that is being used
scalaVersion := "2.11.12"

// libraries that are needed to run ALS
libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.3.2"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.3.2"
libraryDependencies += "org.apache.spark" %% "spark-core" % "2.3.2"
