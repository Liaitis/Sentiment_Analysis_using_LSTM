from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
import re
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import numpy as np
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__)
app.debug = True

DATA_FOLDER = 'data/'

with open('labelencoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

model = load_model('best_model.h5')

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

lemmatizer = WordNetLemmatizer()
CONTRACTION_MAPPING = {
    "ain't": "is not", "aren't": "are not", "can't": "cannot", "cant": "cannot",
    "can't've": "cannot have", "'cause": "because", "could've": "could have",
    "couldn't": "could not", "couldn't've": "could not have", "didn't": "did not",
    "doesn't": "does not", "don't": "do not", "hadn't": "had not",
    "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not",
    "he'd": "he would", "he'd've": "he would have", "he'll": "he will",
    "he'll've": "he will have", "he's": "he is", "how'd": "how did",
    "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
    "I'd": "I would", "I'd've": "I would have", "I'll": "I will",
    "I'll've": "I will have", "I'm": "I am", "I've": "I have",
    "i'd": "i would", "i'd've": "i would have", "i'll": "i will",
    "i'll've": "i will have", "i'm": "i am", "i've": "i have",
    "isn't": "is not", "it'd": "it would", "it'd've": "it would have",
    "it'll": "it will", "it'll've": "it will have", "it's": "it is",
    "let's": "let us", "ma'am": "madam", "mayn't": "may not",
    "might've": "might have", "mightn't": "might not", "mightn't've": "might not have",
    "must've": "must have", "mustn't": "must not", "mustn't've": "must not have",
    "needn't": "need not", "needn't've": "need not have", "o'clock": "of the clock",
    "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
    "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would",
    "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have",
    "she's": "she is", "should've": "should have", "shouldn't": "should not",
    "shouldn't've": "should not have", "so've": "so have", "so's": "so as",
    "this's": "this is", "that'd": "that would", "that'd've": "that would have",
    "that's": "that is", "there'd": "there would", "there'd've": "there would have",
    "there's": "there is", "here's": "here is", "they'd": "they would",
    "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have",
    "they're": "they are", "they've": "they have", "to've": "to have",
    "wasn't": "was not", "we'd": "we would", "we'd've": "we would have",
    "we'll": "we will", "we'll've": "we will have", "we're": "we are",
    "we've": "we have", "weren't": "were not", "what'll": "what will",
    "what'll've": "what will have", "what're": "what are", "what's": "what is",
    "what've": "what have", "when's": "when is", "when've": "when have",
    "where'd": "where did", "where's": "where is", "where've": "where have",
    "who'll": "who will", "who'll've": "who will have", "who's": "who is",
    "who've": "who have", "why's": "why is", "why've": "why have",
    "will've": "will have", "won't": "will not", "won't've": "will not have",
    "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have",
    "y'all": "you all", "y'all'd": "you all would", "y'all'd've": "you all would have",
    "y'all're": "you all are", "y'all've": "you all have", "you'd": "you would",
    "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
    "you're": "you are", "you've": "you have"
}

def decontracted(phrase):
    for word in phrase.split():
        if word.lower() in CONTRACTION_MAPPING:
            phrase = re.sub(r'\b' + re.escape(word) + r'\b', CONTRACTION_MAPPING[word.lower()], phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    phrase = re.sub(r" v", " very", phrase)
    return phrase

def clean_text(text):
    text = decontracted(text)
    replace_white_space = ["\n"]
    for s in replace_white_space:
        text = text.replace(s, " ")

    replace_punctuation = ["’", "‘", "´", "`", "\'", r"\'"]
    for s in replace_punctuation:
        text = text.replace(s, "'")

    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
   
    return ' '.join(lemmatized_tokens)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/single', methods=['POST', 'GET'])
def single():
    if request.method == 'POST':
        text = request.form['text']
        cleaned_text = clean_text(text)
        print(cleaned_text)
        sequence = tokenizer.texts_to_sequences([cleaned_text])
        padded_sequence = pad_sequences(sequence, padding='pre', maxlen=37)
        prediction = model.predict(padded_sequence)
        predicted_label = np.argmax(prediction, axis=1)[0]
        sentiment = encoder.inverse_transform([predicted_label])[0]
        return render_template('single.html', text=cleaned_text, sentiment=sentiment)
    return render_template('single.html')

def predict_sentiment(text):
    cleaned_text = clean_text(text)
    sequences = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(sequences, padding='pre', maxlen=37)
    prediction = model.predict(padded)
    predicted_label = np.argmax(prediction)
    sentiment = encoder.inverse_transform([predicted_label])[0]
    return sentiment

def get_average_sentiment_icon(average_sentiment):
    if average_sentiment > 0:
        return '<i class="fas fa-arrow-up" style="color: green;"></i>'
    elif average_sentiment < 0:
        return '<i class="fas fa-arrow-down" style="color: red;"></i>'
    else:
        return ''

def get_font_color(sentiment):
    if sentiment == 'Positive':
        return 'green'
    elif sentiment == 'Negative':
        return 'red'
    else:
        return 'orange'

@app.route('/batch')
def batch_prediction():
    return render_template('batch.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if not file:
            return render_template('batch.html', error='No file uploaded')
        if file.filename.endswith('.xlsx'):
            df = pd.read_excel(file)
            df['Sentiment'] = df['Review'].apply(predict_sentiment)
            sentiment_df = df[['Name', 'Brand', 'Review', 'Sentiment']]
            sentiment_table = sentiment_df.to_html(classes='data', escape=False, index=False)
            sentiment_table = sentiment_table.replace('<td>', '<td style="color: black;">') 
            for index, row in df.iterrows():
                font_color = get_font_color(row['Sentiment'])
                sentiment_table = sentiment_table.replace(row['Sentiment'], f'<span style="color: {font_color}">{row["Sentiment"]}</span>')

            grouped_df = sentiment_df.groupby(['Name', 'Brand', 'Sentiment']).size().unstack(fill_value=0)
            grouped_df['Total'] = grouped_df.sum(axis=1)
            grouped_df['Positive (%)'] = (grouped_df['Positive'] / grouped_df['Total']) * 100
            grouped_df['Negative (%)'] = (grouped_df['Negative'] / grouped_df['Total']) * 100
            grouped_df['Neutral (%)'] = (grouped_df['Neutral'] / grouped_df['Total']) * 100

            grouped_df['Positive (%)'] = grouped_df['Positive (%)'].apply(lambda x: '{:.2f}'.format(x))
            grouped_df['Negative (%)'] = grouped_df['Negative (%)'].apply(lambda x: '{:.2f}'.format(x))
            grouped_df['Neutral (%)'] = grouped_df['Neutral (%)'].apply(lambda x: '{:.2f}'.format(x))
            grouped_df['Average Sentiment'] = ((grouped_df['Positive'] * 1) + (grouped_df['Neutral'] * 0) + (grouped_df['Negative'] * -1)) / grouped_df['Total']
            grouped_df['Performance'] = grouped_df['Average Sentiment'].apply(get_average_sentiment_icon)
            grouped_df = grouped_df.reset_index()

            grouped_df = grouped_df[['Name', 'Brand', 'Total', 'Positive (%)', 'Negative (%)', 'Neutral (%)', 'Average Sentiment','Performance']]
            grouped_sentiment_table = grouped_df.to_html(index=False, classes='data', escape=False)

            sentiment_df.to_excel(DATA_FOLDER + 'sentiment_table.xlsx', index=False)
            grouped_df.to_excel(DATA_FOLDER + 'grouped_sentiment_table.xlsx', index=False)
            
            return render_template('batch.html', sentiment_table=sentiment_table, grouped_sentiment_table=grouped_sentiment_table)
        else:
            return render_template('batch.html', error='Please upload a valid Excel file')

@app.route('/download_sentiment_table', methods=['POST'])
def download_sentiment_table():
    df = pd.read_excel(DATA_FOLDER + 'sentiment_table.xlsx')
    return send_file(DATA_FOLDER + 'sentiment_table.xlsx', as_attachment=True)

@app.route('/download_grouped_sentiment_table', methods=['POST'])
def download_grouped_sentiment_table():
    df = pd.read_excel(DATA_FOLDER + 'grouped_sentiment_table.xlsx')
    return send_file(DATA_FOLDER + 'grouped_sentiment_table.xlsx', as_attachment=True)

if __name__ == '__main__':
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
    app.run(debug=True)
