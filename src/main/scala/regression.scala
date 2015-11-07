package net.sparktutorials.examples

import org.apache.log4j.{Logger}
//core and SparkSQL
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.sql.DataFrame
// ML Feature Creation, Tuning, Models, and Model Evaluation
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler, OneHotEncoder}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.evaluation.{RegressionEvaluator}
import org.apache.spark.ml.regression.{RandomForestRegressor, LinearRegression}
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.evaluation.RegressionMetrics

object RossmannRegression extends Serializable {
  @transient lazy val logger = Logger.getLogger(getClass.getName)

  val stateHolidayIndexer = new StringIndexer()
    .setInputCol("StateHoliday")
    .setOutputCol("StateHolidayIndex")
  val schoolHolidayIndexer = new StringIndexer()
    .setInputCol("SchoolHoliday")
    .setOutputCol("SchoolHolidayIndex")
  val stateHolidayEncoder = new OneHotEncoder()
    .setInputCol("StateHolidayIndex")
    .setOutputCol("StateHolidayVec")
  val schoolHolidayEncoder = new OneHotEncoder()
    .setInputCol("SchoolHolidayIndex")
    .setOutputCol("SchoolHolidayVec")
  val dayOfMonthEncoder = new OneHotEncoder()
    .setInputCol("DayOfMonth")
    .setOutputCol("DayOfMonthVec")
  val dayOfWeekEncoder = new OneHotEncoder()
    .setInputCol("DayOfWeek")
    .setOutputCol("DayOfWeekVec")
  val storeEncoder = new OneHotEncoder()
    .setInputCol("Store")
    .setOutputCol("StoreVec")

  val assembler = new VectorAssembler()
    .setInputCols(Array("StoreVec", "DayOfWeekVec", "Open",
      "DayOfMonthVec", "StateHolidayVec", "SchoolHolidayVec"))
    .setOutputCol("features")

  def preppedLRPipeline():TrainValidationSplit = {
    val lr = new LinearRegression()
    
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.1, 0.01))
      .addGrid(lr.fitIntercept)
      .addGrid(lr.elasticNetParam, Array(0.0, 0.25, 0.5, 0.75, 1.0))
      .build()

    val pipeline = new Pipeline()
      .setStages(Array(stateHolidayIndexer, schoolHolidayIndexer,
        stateHolidayEncoder, schoolHolidayEncoder, storeEncoder,
        dayOfWeekEncoder, dayOfMonthEncoder,
        assembler, lr))
    
    val tvs = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(new RegressionEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.75)
    tvs
  }

  def preppedRFPipeline():TrainValidationSplit = {
    val dfr = new RandomForestRegressor()

    val paramGrid = new ParamGridBuilder()
      .addGrid(dfr.maxDepth, Array(2, 4, 8, 12))
      .addGrid(dfr.numTrees, Array(20, 50, 100))
      .build()

    val pipeline = new Pipeline()
      .setStages(Array(stateHolidayIndexer, schoolHolidayIndexer,
        stateHolidayEncoder, schoolHolidayEncoder, storeEncoder,
        dayOfWeekEncoder, dayOfMonthEncoder,
        assembler, dfr))

    val tvs = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(new RegressionEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.55)
    tvs
  }

  def fitModel(tvs:TrainValidationSplit, data:DataFrame) = {
    val Array(training, test) = data.randomSplit(Array(0.8, 0.2), seed = 12345)
    logger.info("Fitting data")
    val model = tvs.fit(training)
    logger.info("Now performing test on hold out set")
    val holdout = model.transform(test).select("prediction","label")

    // have to do a type conversion for RegressionMetrics
    val rm = new RegressionMetrics(holdout.rdd.map(x =>
      (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double])))

    logger.info("Test Metrics")
    logger.info("Test Explained Variance:")
    logger.info(rm.explainedVariance)
    logger.info("Test R^2 Coef:")
    logger.info(rm.r2)
    logger.info("Test MSE:")
    logger.info(rm.meanSquaredError)
    logger.info("Test RMSE:")
    logger.info(rm.rootMeanSquaredError)

    model
  }

  def savePredictions(predictions:DataFrame, testRaw:DataFrame, filePath:String) = {
    val tdOut = testRaw
      .select("Id")
      .distinct()
      .join(predictions, testRaw("Id") === predictions("PredId"), "outer")
      .select("Id", "Sales")
      .na.fill(0:Double) // some of our inputs were null so we have to
                         // fill these with something
    tdOut
      .coalesce(1)
      .write.format("com.databricks.spark.csv")
      .option("header", "true")
      .save(filePath)
  }

  def loadTrainingData(sqlContext:HiveContext, filePath:String):DataFrame = {
    val trainRaw = sqlContext
      .read.format("com.databricks.spark.csv")
      .option("header", "true")
      .load(filePath)
      .repartition(30)
    trainRaw.registerTempTable("raw_training_data")

    sqlContext.sql("""SELECT
      double(Sales) label, double(Store) Store, int(Open) Open, double(DayOfWeek) DayOfWeek, 
      StateHoliday, SchoolHoliday, (double(regexp_extract(Date, '\\d+-\\d+-(\\d+)', 1))) DayOfMonth
      FROM raw_training_data
    """).na.drop()
  }

  def loadKaggleTestData(sqlContext:HiveContext, filePath:String) = {
    val testRaw = sqlContext
      .read.format("com.databricks.spark.csv")
      .option("header", "true")
      .load(filePath)
      .repartition(30)
    testRaw.registerTempTable("raw_test_data")

    val testData = sqlContext.sql("""SELECT
      Id, double(Store) Store, int(Open) Open, double(DayOfWeek) DayOfWeek, StateHoliday, 
      SchoolHoliday, (double(regexp_extract(Date, '\\d+-\\d+-(\\d+)', 1))) DayOfMonth
      FROM raw_test_data
      WHERE !(ISNULL(Id) OR ISNULL(Store) OR ISNULL(Open) OR ISNULL(DayOfWeek) 
        OR ISNULL(StateHoliday) OR ISNULL(SchoolHoliday))
    """).na.drop() // weird things happen if you don't filter out the null values manually

    Array(testRaw, testData) // got to hold onto testRaw so we can make sure
    // to have all the prediction IDs to submit to kaggle
  }

  def main(args:Array[String]) = {
    val name = "Linear Regression Application"
    logger.info(s"Starting up $name")
    
    val conf = new SparkConf().setAppName(name)
    val sc = new SparkContext(conf)
    val sqlContext = new HiveContext(sc)
//    sc.setLogLevel("INFO")

    logger.info("Set Up Complete")
    val data = loadTrainingData(sqlContext, args(0))
    val Array(testRaw, testData) = loadKaggleTestData(sqlContext, args(1))

    // The linear Regression Pipeline
    val linearTvs = preppedLRPipeline()
    logger.info("evaluating linear regression")
    val lrModel = fitModel(linearTvs, data)
    logger.info("Generating kaggle predictions")
    val lrOut = lrModel.transform(testData)
      .withColumnRenamed("prediction","Sales")
      .withColumnRenamed("Id","PredId")
      .select("PredId", "Sales")
    savePredictions(lrOut, testRaw, "linear_predictions.csv")

    // The Random Forest Pipeline
    val randomForestTvs = preppedRFPipeline()
    logger.info("evaluating random forest regression")
    val rfModel = fitModel(randomForestTvs, data)
    val rfOut = rfModel.transform(testData)
      .withColumnRenamed("prediction","Sales")
      .withColumnRenamed("Id","PredId")
      .select("PredId", "Sales")
    savePredictions(rfOut, testRaw, "random_forest_predictions.csv")
  }
}
