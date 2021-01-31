package InsuranceClaims
import org.apache.log4j.Logger
object InsuranceClaims extends App {
  val logger = Logger.getLogger(this.getClass)
  logger.info("Starting spark")

  val train = "src/main/resources/InsuranceClaims/train.csv"
  val test = "src/main/resources/InsuranceClaims/test.csv"

  logger.info("Reading data from " + train + " file")
  val trainInput = Extract.readInputData(train)
  logger.info("Reading data from " + test + " file")
  val testInput = Extract.readInputData(test)

  logger.info("Done :) closing spark")
  Extract.spark.close
}
