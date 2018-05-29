# projet_6

This project is part of the openclassrooms.com Data Scientist training.
Description of the subject is https://openclassrooms.com/projects/categorisez-automatiquement-des-questions

It's about automatically proposing tags for a question that is entered in the stackoverflow web site.

Natural language processing uses unsupervised LDA, TD-IDF algorithms, ans a supervised multi-lable SVM. Performance is evaluated for these algorithm and the final recommendation engine is built using SVM.

Please look into docs folder. It contains slides (in french) and and a full report (in english).

All the development has been done in notebooks, which obvioulsy are in the Notebooks folder.
- Text Data exploration has the exploration, feature engineering, test of algorithms
- Text Data-LDA Optimization-Monograms does some optimzation options, including a kind of grid search to find the best number of topics
- Text Data-LDA Optimization does the same but with bigrams and monograms all together
- Text Data-Supervised is the final model based on SVM. This creates the pickle dumps used in the test website.

The website code is TagsReco.py in the python_scripts folder. It was initially developped as a notebook.
