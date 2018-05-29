
# coding: utf-8

# Required libraries
import os
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

import pickle

nltk.download('stopwords')

# Constants
my_data = "../data"

# ## Import data

# Set of legacy questions to train/test the model. Query loads questions where last modification date was last year.
# 
# Stackexchange query is https://data.stackexchange.com/stackoverflow/query/edit/847084
# 
# select Id, Score, ViewCount, CreationDate, LastActivityDate, title, tags, body
# from Posts 
# where (score > 100) and (LastActivityDate > '2017-04-01') 
# and (LastActivityDate < '2018-04-01') and (PostTypeId = 1)

# Load data into Pandas dataframe
datafile = "QueryResultsOld.csv"
full_path = os.path.join(my_data, datafile)
df_questions = pd.read_csv(full_path)
print(df_questions.shape)


# ### Tokenize words

# This piece of code cleans up the HTML content, removes stop words and converts the remaining ones into a list of stems.
# The result is a new column in the table with that list for each question.



# Take only aplanumeric words, no punctuation signs
tokenizer = nltk.RegexpTokenizer('\w+')

# Prepare set of stopwords
stopWords = set(stopwords.words('english'))

# Define stemmer
snowball_stemmer = SnowballStemmer("english")

wordsFiltered = []
wordsArray = []

for html_text in df_questions['body'] + " " + df_questions['title']:
    soup = BeautifulSoup(html_text, "lxml").get_text()
    words = tokenizer.tokenize(soup.lower())
    his_words = ''
    for w in words:
        if w not in stopWords:
            stem = snowball_stemmer.stem(w)
            wordsFiltered.append(stem)
            his_words = his_words + ' ' + stem
    wordsArray.append(his_words)

# Add a column to the dataframe with the list of cleaned stems
df_questions['words'] = wordsArray


# ### Vectorize stems

# The set of stems gets vectorized. Each question is a line of the matrix and each stem is a column. The values are the number of occurrences of each stem in each question.
# Any word that is used in more than 95% of the questions or less than 5 times across all questions is removed, because its either too common or too specific to be used for the topic determination.


stem_vectorizer = CountVectorizer(lowercase=True, ngram_range=(1, 1), max_df=0.95, min_df=5)
stem_matrix = stem_vectorizer.fit_transform(df_questions['words'])


# ### Vectorize tags

# Vectorize all the tags used in the training set
tag_vectorizer = CountVectorizer(lowercase = True, max_df=1.0, min_df=0, token_pattern = '[^<>]+')
tag_matrix = tag_vectorizer.fit_transform(df_questions['tags'])

# Create a datraframe with the list of tags and the number of times they are used
tag_names = tag_vectorizer.get_feature_names()

tag_df = pd.DataFrame(tag_matrix.sum(0)).T
tag_df.rename(index=str, columns={0:'number'}, inplace=True)
tag_df['names'] = tag_names


# ## Supervised methodology

X = stem_matrix
Y = tag_matrix

MultiLabelClassif = OneVsRestClassifier(SVC(kernel='linear', probability=True))
MultiLabelClassif.fit(X, Y)

# dump the trained vectorizer
output = open('../pkl/stem_vectorizer.pkl', 'wb')
pickle.dump(stem_vectorizer, output, -1)
output.close()

# dump the trained classifier
output = open('../pkl/MultiLabelClassif.pkl', 'wb')
pickle.dump(MultiLabelClassif, output, -1)
output.close()

# dump the tags dataframe
output = open('../pkl/tag_df.pkl', 'wb')
pickle.dump(tag_df, output, -1)
output.close()
