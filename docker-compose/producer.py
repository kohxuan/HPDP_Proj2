from kafka import KafkaProducer
import time
import csv
import json

producer = KafkaProducer(bootstrap_servers='localhost:9092')

try:
    with open('Combined_HariRayaComments.csv', 'r', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        for row in reader:
            message = json.dumps(row).encode('utf-8')
            producer.send('youtube-comments', value=message)
            print(f"Sent: {message}")
            time.sleep(0.5)
except Exception as e:
    print(f"Error: {e}")
finally:
    producer.flush()
    producer.close()
