from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return "Sentiment Analysis API Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data["text"]

    vector = vectorizer.transform([text])
    pred = model.predict(vector)[0]

    sentiment = "Positive" if pred == 1 else "Negative"

    return jsonify({"Sentiment": sentiment})

if __name__ == "__main__":
    app.run(debug=True, port=5001)