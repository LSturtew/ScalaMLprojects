package InsuranceClaims

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.DataFrame

object Extract {
  Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
  lazy val spark = SparkWrapper.createSession()
  import spark.implicits._

  def readInputData(filePath:String): DataFrame = {
    spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .format("com.databricks.spark.csv")
      .load(filePath)
      .cache
  }
}
