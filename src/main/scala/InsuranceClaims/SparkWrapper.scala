package InsuranceClaims

import org.apache.spark.sql.SparkSession

object SparkWrapper {
  def createSession(appName:String): SparkSession = {
    SparkSession
      .builder()
      .master("local[*]")
      .appName(appName)
      .getOrCreate()
  }
}