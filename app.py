import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import pad_sequences
from flask import Flask, render_template, request
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly
import plotly.graph_objs as go
import json
import os

app = Flask(__name__)
analyzer = SentimentIntensityAnalyzer()

# Initialize global variables
model = None
tokenizer = None

# Define functions to load model and tokenizer
def load_keras_model():
    global model
    model = load_model('models/uci_sentimentanalysis.h5')

def load_tokenizer():
    global tokenizer
    with open('models/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

# Define sentiment analysis function
def sentiment_analysis(input):
    user_sequences = tokenizer.texts_to_sequences([input])
    user_sequences_matrix = pad_sequences(user_sequences, maxlen=1225)
    prediction = model.predict(user_sequences_matrix)
    return round(float(prediction[0][0]), 2)

# Initialize model and tokenizer at startup
load_keras_model()
load_tokenizer()

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = None
    vader_chart = None
    if request.method == "POST":
        text = request.form.get("user_text")
        sentiment = analyzer.polarity_scores(text)
        sentiment["custom model positive"] = sentiment_analysis(text)

        # Create VADER sentiment chart
        labels = ['Positive', 'Negative', 'Neutral']
        values = [sentiment['pos'], sentiment['neg'], sentiment['neu']]
        colors = ['#27ae60', '#c0392b', '#7f8c8d']

        fig = go.Figure(data=[go.Bar(
            x=labels,
            y=values,
            marker_color=colors
        )])

        fig.update_layout(
            title='VADER Sentiment Scores',
            yaxis_title='Score',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(
                gridcolor='rgba(0,0,0,0.1)',
                range=[0, 1]
            )
        )

        vader_chart = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('form.html', sentiment=sentiment, vader_chart=vader_chart)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
