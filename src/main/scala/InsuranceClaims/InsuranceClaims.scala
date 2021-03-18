package InsuranceClaims

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession


object InsuranceClaims extends App {
  Logger.getLogger("org").setLevel(Level.WARN)
  Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
  val logger = Logger.getLogger(this.getClass)
  logger.info("Starting spark")
  lazy val spark: SparkSession = SparkWrapper.createSession("InsuranceClaims")

  val train = "src/main/resources/InsuranceClaims/train.csv"
  val test = "src/main/resources/InsuranceClaims/test.csv"

  val trainInput = Extract.readInputData(spark, train)
  val trainCount = trainInput.count()
  logger.info("Train data count: " + trainCount)
  val testInput = Extract.readInputData(spark, test)
  val testCount = testInput.count()
  logger.info("Test data count: " + testCount)

  logger.info("Preparing data for training model")
  val data = Extract.renameLabelColumn(trainInput, "loss")
  data.na.drop()
  logger.info("Removed " + (trainCount - data.count()) + " rows containing null values.")

  val (trainingData, validationData) = Extract.splitTrainingSet(data, 12345L, 0.25)
  logger.info("Train data count: " + trainingData.count())
  logger.info("Validation data count: " + validationData.count())
  logger.info("==========================================================================")
  logger.info("Caching the data")

  trainingData.cache
  validationData.cache
  testInput.cache
  val featureColumns = Extract.getFeatureColumns(trainingData.columns)
  logger.info(featureColumns.length + " feature columns selected")
  logger.info("==========================================================================")
  val encoder = Extract.createCategoricalDataEncoder(trainingData.columns, trainInput, testInput)
  val assembler = Extract.createFeatureAssembler(featureColumns)
  logger.info("==========================================================================")
  Model1LinearRegression.train(spark, trainingData, validationData, testInput, encoder, assembler)
  logger.info("==========================================================================")
  Model2GradientBoostTrees.train(spark, trainingData, validationData, testInput, encoder, assembler,featureColumns)
  logger.info("Done :) closing spark")
  spark.close()
}
