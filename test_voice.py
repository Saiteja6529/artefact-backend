import requests
import os

url = "http://127.0.0.1:5000/synthesize"
payload = {"text": "Hello, this is a test."}
response = requests.post(url, json=payload)
if response.status_code == 200:
    with open("test.mp3", "wb") as f:
        f.write(response.content)
    print("Audio saved to test.mp3")
else:
    print(f"Error: {response.status_code} - {response.text}")