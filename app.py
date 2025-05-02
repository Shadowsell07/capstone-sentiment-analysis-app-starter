from flask import Flask, render_template, request
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = Flask(__name__)
analyzer = SentimentIntensityAnalyzer()

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = None
    if request.method == "POST":
        text = request.form.get("user_text")
        sentiment = analyzer.polarity_scores(text)
    return render_template('form.html', sentiment=sentiment)

if __name__ == "__main__":
    app.run()
