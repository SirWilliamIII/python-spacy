import spacy
from spacy import displacy
from flask import Flask, render_template, request, jsonify
from textblob import TextBlob
import en_core_web_sm
from spacy.lang.en.stop_words import STOP_WORDS
from collections import Counter
from string import punctuation
import re
 

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = en_core_web_sm.load()

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text = request.form["text"]
        doc = nlp(text)
        ####################

        
        ######################
        
        # Generate dependency parse
        options = {"compact": True, "bg": "#09a3d5",
           "color": "white", "distance": 90}

        dep_svg = displacy.render(doc, style="dep",page=True, options=options)
        
        # Generate dependency parse
        
        # Generate named entity recognition
        ner_svg = displacy.render(doc, style="ent",page=True, options={"bg": "#ffffff", "color": "#333333", "distance": "90"})
        
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
        
        return render_template("result.html", 
                               dep_svg=dep_svg, 
                               ner_svg=ner_svg, 
                               sentiment=sentiment,
                               word_freq=word_freq,
                               pos_tags=pos_tags,
                               complexity=complexity,
                               summary=summary)
    
    return render_template("index.html")

def get_word_frequency(doc):
    words = [token.text for token in doc if token.is_alpha and token.text.lower() not in STOP_WORDS]
    return Counter(words).most_common(5)

def get_pos_tags(doc):
    return [(token.text, token.pos_) for token in doc]

def get_text_complexity(doc):
    num_words = len([token for token in doc if not token.is_punct])
    num_sentences = len(list(doc.sents))
    avg_word_length = sum(len(token.text) for token in doc if not token.is_punct) / num_words if num_words > 0 else 0
    
    return {
        "num_words": num_words,
        "num_sentences": num_sentences,
        "avg_word_length": round(avg_word_length, 2),
        "readability_score": calculate_readability_score(doc)
    }

def calculate_readability_score(doc):
    # Simple implementation of Flesch-Kincaid Grade Level
    num_words = len([token for token in doc if not token.is_punct])
    num_sentences = len(list(doc.sents))
    num_syllables = sum(count_syllables(token.text) for token in doc if token.is_alpha)
    
    if num_words == 0 or num_sentences == 0:
        return 0
    
    return round(0.39 * (num_words / num_sentences) + 11.8 * (num_syllables / num_words) - 15.59, 2)

def count_syllables(word):
    # Simple syllable counting heuristic
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    return max(count, 1)

def summarize_text(doc):
    # Simple extractive summarization
    keyword = []
    pos_tag = ['PROPN', 'ADJ', 'NOUN', 'VERB']
    for token in doc:
        if(token.text in STOP_WORDS or token.text in punctuation):
            continue
        if(token.pos_ in pos_tag):
            keyword.append(token.text)
    
    freq_word = Counter(keyword)
    max_freq = Counter(keyword).most_common(1)[0][1]
    for word in freq_word.keys():  
        freq_word[word] = (freq_word[word]/max_freq)
        
    sent_strength = {}
    for sent in doc.sents:
        for word in sent:
            if word.text in freq_word.keys():
                if sent in sent_strength.keys():
                    sent_strength[sent] += freq_word[word.text]
                else:
                    sent_strength[sent] = freq_word[word.text]
    
    summarized_sentences = sorted(sent_strength.items(), key=lambda x: x[1], reverse=True)[:2]
    summary = " ".join([str(sentence[0]) for sentence in summarized_sentences])
    
    return summary

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
    app.run(debug=True)
