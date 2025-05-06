from flask import Flask, render_template, request
from textblob import TextBlob
import plotly.graph_objs as go

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment_result = None
    sentiment_data = None
    if request.method == "POST":
        text_input = request.form["text_input"]
        blob = TextBlob(text_input)
        
        # Sentiment analysis
        polarity = blob.sentiment.polarity
        if polarity > 0:
            sentiment_result = "Positive"
        elif polarity < 0:
            sentiment_result = "Negative"
        else:
            sentiment_result = "Neutral"
        
        # Data for the chart
        sentiment_data = {
            "labels": ["Positive", "Neutral", "Negative"],
            "values": [
                max(polarity, 0),        # Positive value
                abs(polarity) <= 0.1 and 1 or 0, # Neutral value
                polarity < 0 and abs(polarity) or 0  # Negative value
            ]
        }

    return render_template("form.html", result=sentiment_result, sentiment_data=sentiment_data)

if __name__ == "__main__":
    app.run(debug=True)
