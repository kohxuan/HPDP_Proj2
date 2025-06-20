from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, lower, regexp_replace, from_json
from pyspark.sql.types import StringType, StructType, StructField
import joblib

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("YouTubeCommentsStreaming") \
    .config("spark.jars.packages", 
        "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.2,"
        "org.elasticsearch:elasticsearch-spark-30_2.12:8.13.4"
    ) \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# Load trained pipeline (TF-IDF + Naive Bayes)
pipeline = joblib.load("/opt/bitnami/spark/work/model/sentiment_model.pkl")
pipeline_broadcast = spark.sparkContext.broadcast(pipeline)

# Kafka JSON schema
schema = StructType([
    StructField("comment_text", StringType(), True),
    StructField("likes", StringType(), True),
    StructField("view_count", StringType(), True)
])

# Define UDF to predict sentiment
def predict_sentiment(text):
    if text and len(text.strip()) > 2:
        try:
            prediction = pipeline_broadcast.value.predict([text])
            return str(prediction[0])
        except Exception as e:
            print(f"Prediction error: {e}")
            return "error"
    return "unknown"

# Register UDF
predict_udf = udf(predict_sentiment, StringType())

# Read streaming data from Kafka
raw_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:29092") \
    .option("subscribe", "youtube-comments") \
    .option("startingOffsets", "latest") \
    .load()

# Convert Kafka value to string and parse JSON
json_df = raw_df.selectExpr("CAST(value AS STRING)")
parsed_df = json_df.select(from_json(col("value"), schema).alias("data")).select("data.*")

# Preprocess text
clean_df = parsed_df.withColumn("clean_text", lower(col("comment_text")))
clean_df = clean_df.withColumn("clean_text", regexp_replace("clean_text", r"http\S+|www\S+|https\S+", ""))
clean_df = clean_df.withColumn("clean_text", regexp_replace("clean_text", r"[^a-zA-Z\s]", ""))

# Predict sentiment
result_df = clean_df.withColumn("sentiment", predict_udf(col("clean_text")))

# Write to console
console_query = result_df.writeStream \
    .outputMode("append") \
    .format("console") \
    .option("truncate", "false") \
    .start()

# Write to Elasticsearch
es_query = result_df.writeStream \
    .outputMode("append") \
    .format("org.elasticsearch.spark.sql") \
    .option("es.nodes", "elasticsearch") \
    .option("es.port", "9200") \
    .option("checkpointLocation", "/tmp/spark_checkpoint") \
    .option("es.resource", "youtube-comments-index") \
    .start()

# Wait for both streams
console_query.awaitTermination()
es_query.awaitTermination()
