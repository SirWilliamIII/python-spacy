import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import os
from spacy import displacy
from flask import Flask, render_template, request, jsonify
from textblob import TextBlob
from wordcloud import WordCloud
from spacy.lang.en.stop_words import STOP_WORDS
from collections import Counter
from string import punctuation
from spacy.language import Language
import re

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load('en_core_web_sm')

app = Flask(__name__)


from helpers.functions import (
    get_word_frequency,
    calc_sentiment_score,
    get_pos_tags,
    get_text_complexity,
    get_lemmatized_text,
    get_lemmatized_score,
    summarize_text
)

@app.route("/", methods=["GET", "POST"])
def index():
    """
    Handle the index route for the Flask application.
    If the request method is POST, this function will:
    - Extract text from the form.
    - Perform natural language processing (NLP) on the text using spaCy.
    - Generate a dependency parse visualization.
    - Generate a named entity recognition (NER) visualization.
    - Perform sentiment analysis using TextBlob.
    - Calculate word frequency.
    - Perform part-of-speech (POS) tagging.
    - Calculate text complexity metrics.
    - Generate a summary of the text.
    - Calculate a lemmatized score.
    - Calculate a sentiment score.
    The results of these analyses are then rendered in the "result.html" template.
    If the request method is not POST, the function renders the "index.html" template.
    Returns:
        Rendered HTML template based on the request method.
    """
    if request.method == "POST":
        text = request.form["text"]
        doc = nlp(text)
        # Generate dependency parse
        options = {"compact": True, "bg": "#09a3d5", "color": "white", "distance": 90}
        dep_svg = displacy.render(doc, style="dep", options=options)
        # Generate named entity recognition
        ner_svg = displacy.render(doc, style="ent", jupyter=False, options={"distance": 90})
        # Perform sentiment analysis
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
        # Get word frequency
        word_freq = get_word_frequency(doc)
        # Get part-of-speech tagging
        pos_tags = get_pos_tags(doc)
        # Get text complexity metrics
        complexity = get_text_complexity(doc)
        # Get text summary
        summary = summarize_text(doc)

        lemmatized_score= get_lemmatized_score(doc)
        
        pos_tag=get_pos_tags(doc)

        sentiment_score=calc_sentiment_score(doc)

        return render_template("result.html", 
                               dep_svg=dep_svg, 
                               ner_svg=ner_svg, 
                               sentiment=sentiment,
                               word_freq=word_freq,
                               pos_tags=pos_tags,
                               complexity=complexity,
                               lemmatized_score=lemmatized_score,
                               pos_tag=pos_tag,
                               calc_sentiment_score=sentiment_score,
                               summary=summary)
    
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.json
    text = data.get("text", "")
    doc = nlp(text)
    
    analysis = {
        "sentiment": TextBlob(text).sentiment.polarity,
        "word_frequency": get_word_frequency(doc),
        "pos_tags": get_pos_tags(doc),
        "complexity": get_text_complexity(doc),
        "summary": summarize_text(doc)
    }
    
    return jsonify(analysis)


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
