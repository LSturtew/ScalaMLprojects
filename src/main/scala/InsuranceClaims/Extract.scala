package InsuranceClaims

import org.apache.log4j.Logger
import org.apache.spark.ml.feature.{StringIndexer, StringIndexerModel, VectorAssembler}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StringType
import org.apache.spark.sql.{DataFrame, SparkSession}

object Extract {
  val logger: Logger = Logger.getLogger(this.getClass)

  def readInputData(spark: SparkSession, filePath: String): DataFrame = {
    logger.info("Reading data from " + filePath + " file")
    spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .format("com.databricks.spark.csv")
      .load(filePath)
      .cache
  }
  def getStringColumnProfile(df: DataFrame, columnName: String): DataFrame = {
    df.select(columnName)
      .withColumn("isEmpty", when(col(columnName) === "", true).otherwise(null))
      .withColumn("isNull", when(col(columnName).isNull, true).otherwise(null))
      .withColumn("fieldLen", length(col(columnName)))
      .agg(
        max(col("fieldLen")).as("maxLength"),
        countDistinct(columnName).as("unique"),
        count("isEmpty").as("isEmpty"),
        count("isNull").as("isNull")
      )
      .withColumn("colName", lit(columnName))
  }

  def renameLabelColumn(trainData: DataFrame, labelColumn: String): DataFrame = {
    logger.info("Renaming column " + labelColumn + " to 'label'")
    trainData.withColumnRenamed(labelColumn, "label")
  }

  def splitTrainingSet(trainData: DataFrame, seed: Long, validationSize: Double): (DataFrame, DataFrame) = {
    logger.info("Split training data into train set and validation set, seed: " + seed + ", validation set size: " + validationSize)
    val splits = trainData.randomSplit(Array(1 - validationSize, validationSize), seed)
    (splits(0), splits(1))
  }

  def getFeatureColumns(columns: Array[String]): Array[String] = {
    logger.info("Get feature columns")
    columns.filter(isNotNeededCategory)
      .filter(isFeature)
      .map(renameCategoryColumns)
  }

  def isNotNeededCategory(column: String): Boolean = !(column matches "cat(109$|110$|112$|113$|116$)")

  def isFeature(column: String): Boolean = !(column matches "id|label")

  def createCategoricalDataEncoder(columns: Array[String], trainData: DataFrame, testData: DataFrame): Array[StringIndexerModel] = {
    logger.info("Create encoder for categorical data")
    columns.filter(isCategory)
      .map(column => new StringIndexer()
        .setInputCol(column)
        .setOutputCol(renameCategoryColumns(column))
        .fit(trainData.select(column).union(testData.select(column))))
  }

  def renameCategoryColumns(column: String): String = if (isCategory(column)) s"idx_$column" else column

  def isCategory(column: String): Boolean = column.startsWith("cat")

  def createFeatureAssembler(featureColumns: Array[String]): VectorAssembler = {
    logger.info("Create feature assembly")
    new VectorAssembler()
      .setInputCols(featureColumns)
      .setOutputCol("features")
  }
}
