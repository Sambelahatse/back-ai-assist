import requests

url = "http://127.0.0.1:5000/ask"
data = {"prompt": "Qui est Mara ?"}

try:
    response = requests.post(url, json=data)
    print("Status Code:", response.status_code)
    print("Response:", response.json())
except Exception as e:
    print("Error:", e)
