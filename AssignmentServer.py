from transformers import pipeline
from fastapi import FastAPI
nlp = pipeline(task='sentiment-analysis',
               model='nlptown/bert-base-multilingual-uncased-sentiment')

app = FastAPI()


@app.get('/')
def get_root():
    return {'message': 'This is the sentiment analysis app'}

@app.get('/sentiment_analysis/')
async def query_sentiment_analysis(text: str):
    return analyze_sentiment(text)

def analyze_sentiment(text):
    """Get and process result"""

    result = nlp(text)

    sent = ''
    if (result[0]['label'] == '1 star'):
        sent = 'very negative'
    elif (result[0]['label'] == '2 star'):
        sent = 'negative'
    elif (result[0]['label'] == '3 stars'):
        sent = 'neutral'
    elif (result[0]['label'] == '4 stars'):
        sent = 'positive'
    else:
        sent = 'very positive'

    prob = result[0]['score']

    # Format and return results
    return {'sentiment': sent, 'probability': prob}
## text preprocessing modules
import requests as r
from string import punctuation
## text preprocessing modules
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re  ## regular expression
import os
from os.path import dirname, join, realpath
import joblib
import uvicorn
from fastapi import FastAPI

app = FastAPI(
    title="Sentiment Model API",
    description="A simple API that use NLP model to predict the sentiment of the movie's reviews",
    version="0.1",
)

## load the sentiment model
with open(
    join(dirname(realpath(__file__)), "models/sentiment_model_pipeline.pkl"), "rb"
) as f:
    model = joblib.load(f)

## cleaning the data
def text_cleaning(text, remove_stop_words=True, lemmatize_words=True):
    ## Clean the text, with the option to remove stop_words and to lemmatize word
    ## Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"http\S+", " link ", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\s+", "", text)  ## remove numbers

    ## Remove punctuation from text
    text = "".join([c for c in text if c not in punctuation])

    ## Optionally, remove stop words
    if remove_stop_words:
        ## load stopwords
        stop_words = stopwords.words("english")
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)

    ## Optionally, shorten words to their stems
    if lemmatize_words:
        text = text.split()
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
        text = " ".join(lemmatized_words)

    ## Return a list of words
    return text

@app.get("/predict-review")
def predict_sentiment(review: str):
    """
    A simple function that receive a review content and predict the sentiment of the content.
    :param review:
    :return: prediction, probabilities
    """
    ## clean the review
    cleaned_review = text_cleaning(review)

    ## perform prediction
    prediction = model.predict([cleaned_review])
    output = int(prediction[0])
    probas = model.predict_proba([cleaned_review])
    output_probability = "{:.2f}".format(float(probas[:, output]))

    ## output dictionary
    sentiments = {0: "Negative", 1: "Positive"}

    ## show results
    result = {"prediction": sentiments[output], "Probability": output_probability}
    return result
