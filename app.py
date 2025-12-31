from flask import Flask, render_template, request
import pickle
from langdetect import detect

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    language = ""

    if request.method == "POST":
        text = request.form["news"]
        language = detect(text)
        vector = vectorizer.transform([text])
        result = model.predict(vector)[0]
        prediction = "FAKE NEWS ❌" if result == "FAKE" else "REAL NEWS ✅"

    return render_template("index.html", prediction=prediction, language=language)

if __name__ == "__main__":
    app.run(debug=True)
