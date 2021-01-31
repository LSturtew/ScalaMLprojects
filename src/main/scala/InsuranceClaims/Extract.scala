package InsuranceClaims

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.DataFrame

object Extract {
  val logger = Logger.getLogger(this.getClass)
  Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
  lazy val spark = SparkWrapper.createSession()
  import spark.implicits._

  def readInputData(filePath:String): DataFrame = {
    logger.info("Reading data from " + filePath + " file")
    spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .format("com.databricks.spark.csv")
      .load(filePath)
      .cache
  }

  def renameLabelColumn(trainData:DataFrame, labelColumn:String): DataFrame = {
    logger.info("Renaming column " + labelColumn + " to 'label'")
    trainData.withColumnRenamed(labelColumn,"label")
  }

  def splitTrainingSet(trainData:DataFrame, seed: Long, validationSize:Double): (DataFrame,DataFrame) = {
    logger.info("Split training data into train set and validation set, seed: "+seed+", validation set size: "+ validationSize)
    val splits = trainData.randomSplit(Array(1-validationSize,validationSize),seed)
    (splits(0),splits(1))
  }
}
