package net.sparktutorials.examples

import org.apache.log4j.{Logger}
//core and SparkSQL
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.sql.DataFrame
// ML Feature Creation, Tuning, Models, and Model Evaluation
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.evaluation.{RegressionEvaluator}
import org.apache.spark.ml.regression.{RandomForestRegressor, LinearRegression}
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.evaluation.RegressionMetrics

object RossmannRegression extends Serializable {
  @transient lazy val logger = Logger.getLogger(getClass.getName)

  // think there's some dirtiness in the data, this works to clean it
  val stateHolidayIndexer = new StringIndexer()
    .setInputCol("StateHoliday")
    .setOutputCol("StateHolidayIndex")

  // think there's some dirtiness in the data, this works to clean it
  val schoolHolidayIndexer = new StringIndexer()
    .setInputCol("SchoolHoliday")
    .setOutputCol("SchoolHolidayIndex")

  val assembler = new VectorAssembler()
    .setInputCols(Array("Store", "DayOfWeek", "Open", "DayOfMonth", "StateHolidayIndex", "SchoolHolidayIndex"))
    .setOutputCol("features")

  def preppedLRPipeline():TrainValidationSplit = {
    val lr = new LinearRegression()
    
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.1, 0.01))
      .addGrid(lr.fitIntercept)
      .addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0))
      .build()

    val pipeline = new Pipeline()
      .setStages(Array(stateHolidayIndexer, schoolHolidayIndexer, assembler, lr))
    
    val tvs = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(new RegressionEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.8)
    tvs
  }

  def preppedRFPipeline():TrainValidationSplit = {
    val dfr = new RandomForestRegressor()
    
    val paramGrid = new ParamGridBuilder()
      .addGrid(dfr.featureSubsetStrategy, Array("auto", "onethird", "sqrt", "log2"))
      .addGrid(dfr.maxBins, Array(5, 15, 25, 35))
      .addGrid(dfr.maxDepth, Array(5, 25, 50, 100))
      .addGrid(dfr.numTrees, Array(5, 25, 50, 100, 250, 500, 1000))
      .build()
    
    val pipeline = new Pipeline()
      .setStages(Array(stateHolidayIndexer, schoolHolidayIndexer, assembler, dfr))
    
    val tvs = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(new RegressionEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.8)
    tvs
  }

  def fitTestEval(tvs:TrainValidationSplit, training:DataFrame, test:DataFrame,
    toPredict:DataFrame):DataFrame = {
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

    logger.info("Generating test predictions")
    model.transform(toPredict)
      .withColumnRenamed("prediction","Sales")
      .select("Id", "Sales")
  }

  def main(args:Array[String]) = {
    val name = "Example Application"
    logger.info(s"Starting up $name")
    
    val conf = new SparkConf().setAppName(name)
    val sc = new SparkContext(conf)
    val sqlContext = new HiveContext(sc)
    sc.setLogLevel("WARN")

    logger.info("Set Up Complete")
    val trainRaw = sqlContext
      .read.format("com.databricks.spark.csv")
      .option("header", "true")
      .load("../mlproject/rossman/train.csv")
    trainRaw.registerTempTable("raw_training_data")

    val data = sqlContext.sql("""SELECT
      double(Sales) label, int(Store) Store, int(Open) Open, int(DayOfWeek) DayOfWeek, 
      StateHoliday, SchoolHoliday, (int(regexp_extract(Date, '\\d+-\\d+-(\\d+)', 1))) DayOfMonth
      FROM raw_training_data
    """).na.drop()

    var rowCount = data.count
    logger.info(s"Row Count for complete training set: $rowCount")
    val Array(training, test) = data.randomSplit(Array(0.8, 0.2), seed = 12345)

    val testRaw = sqlContext
      .read.format("com.databricks.spark.csv")
      .option("header", "true")
      .load("../mlproject/rossman/test.csv")
    testRaw.registerTempTable("raw_test_data")

    val testData = sqlContext.sql("""SELECT
      Id, int(Store) Store, int(Open) Open, int(DayOfWeek) DayOfWeek, StateHoliday, 
      SchoolHoliday, (int(regexp_extract(Date, '\\d+-\\d+-(\\d+)', 1))) DayOfMonth
      FROM raw_test_data
      WHERE !(ISNULL(Id) OR ISNULL(Store) OR ISNULL(Open) OR ISNULL(DayOfWeek) 
        OR ISNULL(StateHoliday) OR ISNULL(SchoolHoliday))
    """).na.drop() // weird things happen if you don't filter out the null values manually

    rowCount = testData.count
    logger.info(s"Row Count for Test: $rowCount")

    // now we move onto the pipeline
    val linearTvs = preppedLRPipeline()
    logger.info("evaluating linear regression")
    training.show()
    val lrOut = fitTestEval(linearTvs, training, test, testData)
    lrOut.show()
      lrOut
        .write.format("com.databricks.spark.csv")
        .option("header", "true")
        .save("linear_regression_predictions.csv")
    logger.info("evaluating random forest regression")
    val treeTvs = preppedRFPipeline()
    val dfrOut = fitTestEval(treeTvs, training, test, testData)
      dfrOut
        .write.format("com.databricks.spark.csv")
        .option("header", "true")
        .save("random_forest_predictions.csv")
  }
}
