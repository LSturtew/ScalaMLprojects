package InsuranceClaims
import org.apache.log4j.Logger
import org.apache.spark.ml.regression.LinearRegressionModel


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
  val encoder = Extract.createCategoricalDataEncoder(trainingData.columns,trainingData,testInput)
  val assembly = Extract.createFeatureAssembly(featureColumns)
  logger.info("==========================================================================")
  val model = Model1LinearRegression.createModel()
  val pipeline = Model1LinearRegression.createPipeline(encoder,assembly,model)
  val paramGrid = Model1LinearRegression.createParameterGrid(model)
  val crossVal = Model1LinearRegression.createCrossValidator(pipeline,paramGrid)
  val fittedModel = Model1LinearRegression.fitModel(trainingData,crossVal)
  val trainPredictions = Model1LinearRegression.evaluateModel(trainingData,fittedModel)
  val trainRegressionMetrics = Model1LinearRegression.createRegressionMetrics(trainPredictions)
  val validationPredictions = Model1LinearRegression.evaluateModel(validationData,fittedModel)
  val validationRegressionMetrics = Model1LinearRegression.createRegressionMetrics(validationPredictions)
  logger.info("==========================================================================")
  val bestModel = Model1LinearRegression.getBestModel(fittedModel)
  Model1LinearRegression.predict(fittedModel,testInput)
  logger.info("Done :) closing spark")
  Extract.spark.close
}
