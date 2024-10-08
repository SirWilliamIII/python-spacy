from collections import Counter
from textblob import TextBlob
from string import punctuation

STOP_WORDS = set()  # Define your stop words here

def get_word_frequency(doc):
    words = [token.text for token in doc if token.is_alpha and token.text.lower() not in STOP_WORDS]
    return Counter(words).most_common(5)

def calc_sentiment_score(doc):
    blob = TextBlob(doc.text)
    return f"Polarity: {blob.sentiment.polarity}, Subjectivity: {blob.sentiment.subjectivity}"

def get_pos_tags(doc):
    parts_of_speech = [(token.text, token.pos_) for token in doc]
    pos_tag = ['PROPN', 'ADJ', 'NOUN', 'VERB', 'PUNCT', 'NUM', 'ADV', 'PRON', 'ADP', 'DET', 'AUX', 'CCONJ', 'SPACE', 'PART']

    pos_counts = {}
    sorted_tags = []
    
    for tag, words in parts_of_speech:
        if words in pos_tag:
            if words in pos_counts:
                pos_counts[words] += 1
            else:
                pos_counts[words] = 1
            tags = [f"{pos}: {count}" for pos, count in pos_counts.items()]
            sorted_tags = sorted(tags, key=lambda x: int(x.split(": ")[1]), reverse=True)
    
    return sorted_tags

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
    num_words = len([token for token in doc if not token.is_punct])
    num_sentences = len(list(doc.sents))
    num_syllables = sum(count_syllables(token.text) for token in doc if token.is_alpha)
    
    if num_words == 0 or num_sentences == 0:
        return 0
    
    return round(0.39 * (num_words / num_sentences) + 11.8 * (num_syllables / num_words) - 15.59, 2)

def get_lemmatized_text(doc):
    for token in doc:
        print(token.text, '\t', token.pos_, '\t', token.lemma, '\t', token.lemma_)
    return " ".join([token.lemma_ for token in doc if token])

def get_lemmatized_score(doc):
    lemmatized_words = sum(token.text != token.lemma_ for token in doc if token.is_alpha)
    total_words = sum(1 for token in doc if token.is_alpha)
    score = (lemmatized_words / total_words) * 100 if total_words > 0 else 0
    return(f"Lemmatized score: {str(round(score, 3))}  Lemmatized words: {str(lemmatized_words)} Total Words: {str(total_words)}")

def count_syllables(word):
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
    keywords = extract_keywords(doc)
    freq_word = calculate_word_frequencies(keywords)
    sent_strength = calculate_sentence_strength(doc, freq_word)
    summary = generate_summary(sent_strength)
    return summary

def extract_keywords(doc):
    keywords = []
    pos_tag = ['PROPN', 'ADJ', 'NOUN', 'VERB']
    for token in doc:
        if token.text not in STOP_WORDS and token.text not in punctuation and token.pos_ in pos_tag:
            keywords.append(token.text)
    return keywords

def calculate_word_frequencies(keywords):
    freq_word = Counter(keywords)
    max_freq = freq_word.most_common(1)[0][1]
    for word in freq_word.keys():
        freq_word[word] = freq_word[word] / max_freq
    return freq_word

def calculate_sentence_strength(doc, freq_word):
    sent_strength = {}
    for sent in doc.sents:
        for word in sent:
            if word.text in freq_word.keys():
                if sent in sent_strength.keys():
                    sent_strength[sent] += freq_word[word.text]
                else:
                    sent_strength[sent] = freq_word[word.text]
    return sent_strength

def generate_summary(sent_strength):
    summarized_sentences = sorted(sent_strength.items(), key=lambda x: x[1], reverse=True)[:2]
    summary = " ".join([str(sentence[0]) for sentence in summarized_sentences])
    return summary