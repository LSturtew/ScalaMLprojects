name := "InsuranceClaims"

version := "0.1"

scalaVersion := "2.12.10"

libraryDependencies ++= Seq(
  // https://mvnrepository.com/artifact/org.apache.spark/spark-core
  "org.apache.spark" %% "spark-core" % "2.4.5",
  "org.apache.spark" %% "spark-sql" % "2.4.5",
  "org.apache.spark" %% "spark-mllib" % "2.4.5",
  "org.scalatest" %% "scalatest" % "3.0.5" % Test,
  "com.github.mrpowers" %% "spark-fast-tests" % "0.21.3" % "test",
  "org.apache.logging.log4j" % "log4j-api" % "2.13.3",
  "org.apache.logging.log4j" % "log4j-core" % "2.13.3",

)
