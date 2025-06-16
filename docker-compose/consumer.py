from kafka import KafkaConsumer

# Create Kafka consumer
consumer = KafkaConsumer(
    'youtube-comments',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',
    group_id='comment-consumers'
)

# Read messages
for message in consumer:
    print(f"Received: {message.value.decode('utf-8')}")
