package net.sparktutorials.examples

import org.apache.log4j.{Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.hive.HiveContext

import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler

import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.Pipeline

import org.apache.spark.mllib.evaluation.RegressionMetrics


object RossmanLinearRegression extends Serializable {
  @transient lazy val logger = Logger.getLogger(getClass.getName)

  def convertColumns(df: org.apache.spark.sql.DataFrame, colTypeMap: Map[String, String]) = {
    var localDf = df
    for (Tuple2(column, newType) <- colTypeMap.iterator) {
      localDf = localDf.withColumn(column, localDf.col(column).cast(newType))
    }
    localDf
  }

  def preppedPipeline():TrainValidationSplit = {

    val indexer = new StringIndexer()
      .setInputCol("Date")
      .setOutputCol("DateIndex")

    val assembler = new VectorAssembler()
      .setInputCols(Array("Store", "DayOfWeek", "Customers", "Open", "DateIndex"))
      .setOutputCol("features")

    val lr = new LinearRegression()
    
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.1, 0.01))
      .addGrid(lr.fitIntercept)
      .addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0))
      .build()

    val pipeline = new Pipeline()
      .setStages(Array(indexer, assembler, lr))
    
    val tvs = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(new RegressionEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.8)
    tvs
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
    var data = convertColumns(trainRaw,
      Map("Sales" -> "Double", "Store" -> "Int",
        "DayOfWeek" -> "Int", "Open" -> "Int"))
    data = data.withColumnRenamed("Sales","label")

    var rowCount = data.count
    logger.info(s"Row Count for complete training set: $rowCount")
    val Array(training, test) = data.randomSplit(Array(0.8, 0.2), seed = 12345)

    val testRaw = sqlContext
      .read.format("com.databricks.spark.csv")
      .option("header", "true")
      .load("../mlproject/rossman/test.csv")

    val testData = convertColumns(testRaw,
      Map("Store" -> "Int", "DayOfWeek" -> "Int",
        "Customers" -> "Int", "Open" -> "Int"))

    rowCount = testData.count
    logger.info(s"Row Count for Test: $rowCount")

    // now we move onto the pipeline
    val tvs = preppedPipeline()
    logger.info("Fitting data")
    val model = tvs.fit(training)
    logger.info("Now performing test on hold out set")
    val holdout = model.transform(test).select("prediction","label")

    // have to do a type conversion for Regression
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
    val predicted = model.transform(testData)
      .withColumnRenamed("prediction","Sales")
      .select("Index", "Sales")
      .write.format("com.databricks.spark.csv")
      .save("predicted.csv")
  }
}
