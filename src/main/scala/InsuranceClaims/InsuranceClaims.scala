package InsuranceClaims
import org.apache.log4j.Logger
object InsuranceClaims extends App {
  val logger = Logger.getLogger(this.getClass)
  logger.info("Starting spark")

  val train = "src/main/resources/InsuranceClaims/train.csv"
  val test = "src/main/resources/InsuranceClaims/test.csv"


  val trainInput = Extract.readInputData(train)
  val trainCount =  trainInput.count()
  logger.info("Train data count: " + trainCount)
  val testInput = Extract.readInputData(test)
  val testCount = testInput.count()
  logger.info("Test data count: " + testCount)

  logger.info("Preparing data for training model")
  val data = Extract.renameLabelColumn(trainInput,"loss")
  data.na.drop()
  logger.info("Removed " + (trainCount - data.count()) + " rows containing null values.")

  val (trainingData, validationData) = Extract.splitTrainingSet(data,12345L,0.25)

  logger.info("Caching the data")
  trainingData.cache
  validationData.cache
  testInput.cache

  val featureColumns = Extract.getFeatureColumns(trainingData.columns)
  logger.info(featureColumns.length + " feature columns selected")

  val encoder = Extract.createCategoricalDataEncoder(trainingData.columns,trainingData,testInput)
  val assembly = Extract.createFeatureAssembly(featureColumns)

  logger.info("Done :) closing spark")
  Extract.spark.close
}
