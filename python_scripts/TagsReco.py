
# coding: utf-8

# Required libraries

from flask import Flask, request
import pickle

from nltk.stem import SnowballStemmer
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup

my_home = "/home/muths/p6"
# my_home = ".."
my_data_p6 = my_home + '/data'
my_html_p6 = my_home + '/html'

# Load objects from Pickle
stem_file = open(my_home + '/pkl/stem_vectorizer.pkl', 'rb')
stem_vectorizer = pickle.load(stem_file)
stem_file.close()

classifier_file = open(my_home + '/pkl/MultiLabelClassif.pkl', 'rb')
MultiLabelClassif = pickle.load(classifier_file)
classifier_file.close()

tags_file = open(my_home + '/pkl/tag_df.pkl', 'rb')
tag_df = pickle.load(tags_file)
tags_file.close()


# Function to return the recommended tags
def predict_tags(title=None, body=None):
    # Take only aplanumeric words, no punctuation signs
    tokenizer = RegexpTokenizer('\w+')

    # Prepare set of stopwords
    stop_words = set(stopwords.words('english'))

    # Define stemmer
    snowball_stemmer = SnowballStemmer("english")
    
    new_words_array = []

    html_text = title + " " + body
    soup = BeautifulSoup(html_text, "lxml").get_text()
    words = tokenizer.tokenize(soup.lower())
    his_words = ''
    for w in words:
        if w not in stop_words:
            stem = snowball_stemmer.stem(w)
            his_words = his_words + ' ' + stem
    new_words_array.append(his_words)

    # Vectorize stems using trained vectorizer
    stem_vector_new = stem_vectorizer.transform(new_words_array)
    
    # Predic proba for each tag to be relevant for the question
    new_predict_proba = MultiLabelClassif.predict_proba(stem_vector_new)
    
    proposed_tags = ''
    for i in new_predict_proba[0].argsort()[:-10-1:-1]:
        proposed_tag = tag_df.iloc[i]['names']
        if new_predict_proba[0][i] > 0.5:
            proposed_tag = "<b>" + proposed_tag + "</b>"
        proposed_tags += proposed_tag + "<br/>"
        
    return proposed_tags


app = Flask(__name__)


@app.route('/p6/input')
def quesion():

    # Build input form for questions
    body = ""
   
    # Read HTML header and footer files
    with open(my_html_p6 + '/header.html', 'r') as header:
        html_header = header.read()
    with open(my_html_p6 + '/footer.html', 'r') as footer:
        html_footer = footer.read()
    return html_header + body + html_footer


@app.route('/p6/tag_reco', methods=['GET', 'POST'])
def tag_reco():
    title = request.form['title']
    body = request.form['body']
    proposed_tags = predict_tags(title, body)
    
    reminder = "<h2>Your question</h2>"
    reminder += "<h3>" + title + "</h3>"
    reminder += body
    
    # Read HTML header and footer files
    with open(my_html_p6 + '/header_resp.html', 'r') as header:
        html_header = header.read()
    with open(my_html_p6 + '/footer_resp.html', 'r') as footer:
        html_footer = footer.read()
    return html_header + proposed_tags + reminder + html_footer


if __name__ == "__main__":

    app.run()
