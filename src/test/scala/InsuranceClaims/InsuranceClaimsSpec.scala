package InsuranceClaims

import org.scalatest.FunSpec
import com.github.mrpowers.spark.fast.tests.DataFrameComparer
class InsuranceClaimsSpec extends FunSpec with SparkSessionTestWrapper with DataFrameComparer {
  val file_path = "src/test/resources/InsuranceClaims/train.csv"

  val data = Extract.readInputData(file_path)
  it ("creating train data from csv file"){
    assert(data.count() === 5)
  }

  it("renames the label column"){
    val expected = data.withColumnRenamed("loss","label")
    val result = Extract.renameLabelColumn(data,"loss")
    assertSmallDataFrameEquality(result, expected)
  }

  it("splits the training data into a train set and validation set"){
    val (trainingData, validationData) = Extract.splitTrainingSet(data,12345L,0.2)
    assert(trainingData.count()+validationData.count() == data.count())
  }
}