package InsuranceClaims

import org.apache.log4j.Logger
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{StringIndexerModel, VectorAssembler}
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

import java.nio.file.{FileSystems, Files}


object Model1LinearRegression {

  val logger:Logger = Logger.getLogger(this.getClass)

  val numFolds = 10
  val maxIter: Seq[Int] = Seq(1000)
  val regParam: Seq[Double] = Seq(0.001)
  val tol: Seq[Double] = Seq(1e-6)
  val elasticNetParam:Seq[Double] = Seq(0.001)

  def train(spark:SparkSession,
            trainData: DataFrame,
            validationData:DataFrame,
            testData:DataFrame,
            encoder:Array[StringIndexerModel],
            assembly: VectorAssembler):Unit={
    import spark.implicits._
    logger.info("Creating a linear regression model")
    val model = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("label")

    logger.info("Creating linear regression pipeline")
    val pipeline = new Pipeline()
      .setStages((encoder:+assembly):+model)

    logger.info("Creating parameter grid")
    logger.info("Param maxIter: "+ maxIter)
    val paramGrid = new ParamGridBuilder()
      .addGrid(model.maxIter, maxIter)
      .addGrid(model.regParam,regParam)
      .addGrid(model.tol,tol)
      .addGrid(model.elasticNetParam,elasticNetParam)
      .build()

    logger.info("Creating cross validator")
    logger.info("Param numFolds: " + numFolds)
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new RegressionEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(numFolds)

    def fitModel(trainingData:DataFrame, cv:CrossValidator): CrossValidatorModel ={
      val path:String = "model/InsuranceClaims/Model1LinearRegression"
      if (Files.exists(FileSystems.getDefault.getPath(path))) {
        logger.info("Loading linear regression model")
        CrossValidatorModel.load(path)
      } else {
        logger.info("Fit linear regression algorithm")
        val fitted = cv.fit(trainingData)
        logger.info("Saving model")
        fitted.write.overwrite().save(path)
        fitted
      }

    }

    val fittedModel = fitModel(trainData,cv)

    def evaluateModel(data:DataFrame, model:CrossValidatorModel): RegressionMetrics = {
      logger.info("Evaluating model")
      val predictions = model.transform(data)
        .select("label", "prediction")
        .map { case Row(label: Double, prediction: Double) => (label, prediction) }.rdd
      val rm = new RegressionMetrics(predictions)
      logger.info("Explained Variance: " + rm.explainedVariance)
      logger.info("R^2 Coefficient: " + rm.r2)
      logger.info("MSE: " + rm.meanSquaredError)
      logger.info("RMSE: " + rm.rootMeanSquaredError)
      rm
    }

    logger.info("Metrics from training data")
    evaluateModel(trainData,fittedModel)
    logger.info("Metrics from validation data")
    evaluateModel(validationData,fittedModel)

    logger.info("Get best model")
    val bestModel = fittedModel.bestModel.asInstanceOf[PipelineModel]

    logger.info("CV params explained: "+ fittedModel.explainParams)
    logger.info("GBT params explained: "+ bestModel.stages.last.asInstanceOf[LinearRegressionModel].explainParams)

    logger.info("Predicting loss for test data and saving results")
    fittedModel.transform(testData)
      .select("id","prediction")
      .withColumnRenamed("prediction","loss")
      .coalesce(1)
      .write.format("com.databricks.spark.csv")
      .mode("overwrite")
      .option("header","true")
      .save("output/InsuranceClaims/resultLinearRegression.csv")
  }

}
