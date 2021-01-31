package InsuranceClaims

import org.apache.log4j.Logger
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{StringIndexerModel, VectorAssembler}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

import java.nio.file.{FileSystems, Files}


object Model1LinearRegression {
  lazy val spark:SparkSession = SparkWrapper.createSession()
  import spark.implicits._
  val logger:Logger = Logger.getLogger(this.getClass)
  val numFolds = 10
  val maxIter: Seq[Int] = Seq(1000)
  val regParam: Seq[Double] = Seq(0.001)
  val tol: Seq[Double] = Seq(1e-6)
  val elasticNetParam:Seq[Double] = Seq(0.001)

  def createModel():LinearRegression ={
    logger.info("Creating a linear regression model")
    new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("label")
  }

  def createPipeline(encoder:Array[StringIndexerModel], assembly: VectorAssembler, model:LinearRegression):Pipeline = {
    logger.info("Creating linear regression pipeline")
    new Pipeline()
      .setStages((encoder:+assembly):+model)
  }

  def createParameterGrid(model:LinearRegression):Array[ParamMap] ={
    logger.info("Creating parameter grid")
    logger.info("Param maxIter: "+ maxIter)
    new ParamGridBuilder()
      .addGrid(model.maxIter, maxIter)
      .addGrid(model.regParam,regParam)
      .addGrid(model.tol,tol)
      .addGrid(model.elasticNetParam,elasticNetParam)
      .build()
  }

  def createCrossValidator(pipeline: Pipeline, paramGrid:Array[ParamMap]): CrossValidator ={
    logger.info("Creating cross validator")
    logger.info("Param numFolds: " + numFolds)
    new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new RegressionEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(numFolds)
  }

  def fitModel(trainingData:DataFrame, cv:CrossValidator): CrossValidatorModel ={
    val path:String = "model/InsuranceClaims/Model1LinearRegression"
    if (Files.exists(FileSystems.getDefault.getPath(path))) {
      logger.info("Loading linear regression model")
      CrossValidatorModel.load(path)
    } else {
      logger.info("Fit linear regression algorithm")
      val fitted = cv.fit(trainingData)
      logger.info("CV params explained: " + fitted.explainParams)
      logger.info("Saving model")
      fitted.write.overwrite().save(path)
      fitted
    }
  }

  def evaluateModel(data:DataFrame, model:CrossValidatorModel): RDD[(Double, Double)] = {
    logger.info("Evaluating model")
    model.transform(data)
      .select("label", "prediction")
      .map { case Row(label: Double, prediction: Double) => (label, prediction) }.rdd
  }

  def createRegressionMetrics(predictions:RDD[(Double,Double)]): RegressionMetrics ={
    logger.info("Calculating regression metrics")
    new RegressionMetrics(predictions)
  }

  def getBestModel(model:CrossValidatorModel): PipelineModel ={
    logger.info("Get best model")
    model.bestModel.asInstanceOf[PipelineModel]
  }

  def predict(model:CrossValidatorModel, data:DataFrame): Unit ={
    logger.info("Predicting loss for test data and saving results")
    model.transform(data)
      .select("id","prediction")
      .withColumnRenamed("prediction","loss")
      .coalesce(1)
      .write.format("com.databricks.spark.csv")
      .mode("overwrite")
      .option("header","true")
      .save("output/InsuranceClaims/resultLinearRegression.csv")
  }

}
