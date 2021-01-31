package InsuranceClaims

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.{StringIndexer, StringIndexerModel, VectorAssembler}
import org.apache.spark.sql.{DataFrame, SparkSession}

object Extract {
  val logger:Logger = Logger.getLogger(this.getClass)
  Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
  lazy val spark:SparkSession = SparkWrapper.createSession()

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

  def isCategory(column: String): Boolean = column.startsWith("cat")

  def renameCategoryColumns(column:String):String = if (isCategory(column)) s"idx_$column" else column

  def removeLatsCategories(column: String): Boolean = !(column matches "cat(109$|110$|112$|113$|116$)")

  def onlyFeatureColumns(column: String): Boolean = !(column matches "id|label")

  def getFeatureColumns(columns: Array[String]): Array[String] ={
    logger.info("Get feature columns")
    columns.filter(removeLatsCategories)
      .filter(onlyFeatureColumns)
      .map(renameCategoryColumns)
  }

  def createCategoricalDataEncoder(columns:Array[String], trainData:DataFrame,testData:DataFrame): Array[StringIndexerModel]={
    logger.info("Create encoder for categorical data")
    columns.filter(isCategory)
      .map(column => new StringIndexer()
      .setInputCol(column)
      .setOutputCol(renameCategoryColumns(column))
      .fit(trainData.select(column).union(testData.select(column))))
  }

  def createFeatureAssembly(featureColumns:Array[String]): VectorAssembler ={
    logger.info("Create feature assembly")
    new VectorAssembler()
      .setInputCols(featureColumns)
      .setOutputCol("features")
  }
}
