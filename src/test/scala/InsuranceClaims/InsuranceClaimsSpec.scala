package InsuranceClaims

import org.scalatest.FunSpec
import com.github.mrpowers.spark.fast.tests.DataFrameComparer
import org.apache.spark.sql.DataFrame

class InsuranceClaimsSpec extends FunSpec with SparkSessionTestWrapper with DataFrameComparer {
  val filePathTrain = "src/test/resources/InsuranceClaims/train.csv"
  val filePathTest = "src/test/resources/InsuranceClaims/test.csv"

  val trainData: DataFrame = Extract.readInputData(filePathTrain)
  val testData: DataFrame = Extract.readInputData(filePathTest)

  import spark.implicits._

  describe("Training data"){
    it ("should have 5 rows"){
      assert(trainData.count() === 5)
    }
    it ("should have 132 columns"){
      assert(trainData.columns.length equals 132)
    }
    it ("should have a 'label' column"){
      val expected = trainData.withColumnRenamed("loss","label")
      val result = Extract.renameLabelColumn(trainData,"loss")
      assertSmallDataFrameEquality(result, expected)
    }
    it ("should have the same number of rows before and after splitting"){
      val (trainingData, validationData) = Extract.splitTrainingSet(trainData,12345L,0.2)
      assert(trainingData.count()+validationData.count() == trainData.count())
    }
  }

  describe("Feature columns"){
    it("should return true for categorical columns"){
      val column = "cat1"
      assert(Extract.isCategory(column))
    }
    it("should return false for non-categorical columns"){
      val column = "cont1"
      assert(!Extract.isCategory(column))
    }
    it("should return a new name for categorical columns"){
      val column = "cat1"
      val expected = "idx_cat1"
      assert(Extract.renameCategoryColumns(column)==expected)
    }
    it("should return the same name for non-categorical columns"){
      val column = "id"
      val expected = "id"
      assert(Extract.renameCategoryColumns(column)==expected)
    }
    it("should return true for column cat1"){
      val column = "cat1"
      assert(Extract.isNotNeededCategory(column))
    }
    it("should return false for column cat109"){
      val column = "cat109"
      assert(!Extract.isNotNeededCategory(column))
    }
    it("should return true for column cat2"){
      val column = "cat2"
      assert(Extract.isFeature(column))
    }
    it("should return false for column id"){
      val column = "id"
      assert(!Extract.isFeature(column))
    }
    it("should return false for column label"){
      val column = "label"
      assert(!Extract.isFeature(column))
    }
    it("should return 125 feature columns"){
      val columns = Extract.getFeatureColumns(Extract.renameLabelColumn(trainData,"loss").columns)
      assert(columns.length equals 125)
    }

  }
  describe{"Pipeline"}{
    it("should convert categorical data into numerical data"){
      val encoder = Extract.createCategoricalDataEncoder(Array("cat1"), trainData,testData)
      val encoded = encoder(0).transform(trainData)
      val expected = Seq(0.0,0.0,0.0,1.0,0.0).toDF("idx_cat1")
      assertSmallDataFrameEquality(encoded.select("idx_cat1"),expected)
    }
    it("should create a feature vector"){
      val assembler = Extract.createFeatureAssembly(Array("cont1","cont2","cont3","cont4"))
      val output = assembler.transform(trainData)
      assert(output.columns.contains("features"))
    }
  }
}