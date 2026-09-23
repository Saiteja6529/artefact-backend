import requests

url = "http://127.0.0.1:5000/synthesize"
payload = {"text": "Hello"}

try:
    response = requests.post(url, json=payload)
    print(f"Status code: {response.status_code}")
    if response.status_code == 200:
        print("✅ Route works!")
    else:
        print(f"Response: {response.text[:200]}")
except Exception as e:
    print(f"Error: {e}")