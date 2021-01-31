package InsuranceClaims

import org.scalatest.FunSpec
import com.github.mrpowers.spark.fast.tests.DataFrameComparer
class InsuranceClaimsSpec extends FunSpec with SparkSessionTestWrapper with DataFrameComparer {

  it ("creating train data from csv file"){
    val file_path = "src/test/resources/InsuranceClaims/train.csv"

    val data = Extract.readInputData(file_path)
    assert(data.count() === 5)
  }
}