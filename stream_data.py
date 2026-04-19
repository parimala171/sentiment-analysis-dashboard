import pandas as pd
import requests
import time

df = pd.read_csv("cleaned_data.csv")

for i, row in df.sample(20).iterrows():
    data = {"text": row["text"]}

    try:
        res = requests.post("http://127.0.0.1:5000/predict", json=data)
        print(res.json())
    except:
        print("API not running")

    time.sleep(2)