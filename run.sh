currdir=$(pwd)
jarfile=$(eval find $currdir/target/ -name *.jar)
class=$1
passThroughArgs=${@:2}

echo "Jar File Being Used: $jarfile"
echo "Spark home: $SPARK_HOME"
if [ ! -d "$SPARK_HOME" ]
then
    echo "The SPARK_HOME environmental variable is not set"
    echo "Please set the SPARK_HOME variable"
    exit 1
fi
rm -rf linear_regression_predictions.csv
rm -rf random_forest_predictions.csv
sbt package && $SPARK_HOME/bin/spark-submit \
                   --packages com.databricks:spark-csv_2.11:1.1.0 \
                   --class net.sparktutorials.examples.$class \
                   $jarfile $passThroughArgs
