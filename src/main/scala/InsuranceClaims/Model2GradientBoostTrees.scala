package InsuranceClaims

import org.apache.log4j.Logger
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{StringIndexerModel, VectorAssembler}
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor, LinearRegressionModel}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

import java.nio.file.{FileSystems, Files}

object Model2GradientBoostTrees {

  val numTrees: Seq[Int] = Seq(5, 10, 15)
  val maxBins: Seq[Int] = Seq(32)
  val numFolds: Int = 10
  val maxIter: Seq[Int] = Seq(10)
  val maxDepth: Seq[Int] = Seq(10)

  def train(spark: SparkSession,
            trainData: DataFrame,
            validationData: DataFrame,
            testData: DataFrame,
            encoder: Array[StringIndexerModel],
            assembler: VectorAssembler,
            featureColumns: Array[String]): Unit = {
    import spark.implicits._
    val logger: Logger = Logger.getLogger(this.getClass)

    logger.info("Creating a gradient boost trees model")
    val model = new GBTRegressor()
      .setFeaturesCol("features")
      .setLabelCol("label")

    logger.info("Creating linear regression pipeline")
    val pipeline = new Pipeline()
      .setStages((encoder :+ assembler) :+ model)

    logger.info("Creating parameter grid")
    logger.info("Param maxIter: " + maxIter)
    val paramGrid = new ParamGridBuilder()
      .addGrid(model.maxIter, maxIter)
      .addGrid(model.maxDepth, maxDepth)
      .addGrid(model.maxBins, maxBins)
      .build()

    logger.info("Creating cross validator")
    logger.info("Param numFolds: " + numFolds)
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new RegressionEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(numFolds)

    def fitModel(trainingData: DataFrame, cv: CrossValidator): CrossValidatorModel = {
      val path: String = "model/InsuranceClaims/Model2GradientBoostTrees"
      if (Files.exists(FileSystems.getDefault.getPath(path))) {
        logger.info("Loading GBT model")
        CrossValidatorModel.load(path)
      } else {
        logger.info("Fit GBT algorithm")
        val fitted = cv.fit(trainingData)
        logger.info("Saving model")
        fitted.write.overwrite().save(path)
        fitted
      }
    }

    val fittedModel = fitModel(trainData, cv)

    def evaluateModel(data: DataFrame, model: CrossValidatorModel): RegressionMetrics = {
      logger.info("Evaluating model")
      val predictions = model.transform(data)
        .select("label", "prediction")
        .map { case Row(label: Double, prediction: Double) => (label, prediction) }.rdd
      val rm = new RegressionMetrics(predictions)
      logger.info("Explained Variance: " + rm.explainedVariance)
      logger.info("R^2 Coefficient: " + rm.r2)
      logger.info("MSE: " + rm.meanSquaredError)
      logger.info("MAE: " + rm.meanAbsoluteError)
      logger.info("RMSE: " + rm.rootMeanSquaredError)
      rm
    }

    logger.info("Metrics from training data")
    evaluateModel(trainData, fittedModel)
    logger.info("Metrics from validation data")
    evaluateModel(validationData, fittedModel)

    logger.info("Get best model")
    val bestModel = fittedModel.bestModel.asInstanceOf[PipelineModel]

    val featureImportance = bestModel.stages.last.asInstanceOf[GBTRegressionModel].featureImportances.toArray

    val sortedFeatureList = featureImportance.toList.sorted.toArray

    logger.info("CV params explained: " + fittedModel.explainParams)
    logger.info("GBT params explained: " + bestModel.stages.last.asInstanceOf[GBTRegressionModel].explainParams)
    logger.info("GBT features explained: " + featureColumns.zip(sortedFeatureList).map(t => s"\t${t._1} = ${t._2}").mkString("\n"))

    logger.info("Predicting loss for test data and saving results")
    fittedModel.transform(testData)
      .select("id", "prediction")
      .withColumnRenamed("prediction", "loss")
      .coalesce(1)
      .write.format("com.databricks.spark.csv")
      .mode("overwrite")
      .option("header", "true")
      .save("output/InsuranceClaims/resultGradientBoostTrees.csv")

  }

}
