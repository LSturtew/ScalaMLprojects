package InsuranceClaims

import org.apache.spark.sql.SparkSession

object SparkWrapper {
  def createSession(): SparkSession = {
    SparkSession
      .builder()
      .master("local[*]")
      .appName("InsuranceClaims")
      .getOrCreate()
  }
}