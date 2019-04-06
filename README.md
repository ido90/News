# News
Scrapping and some analysis of several popular Israeli news websites.

Due to the choice to work with Hebrew text, most of the state-of-the-art of NLP conventional processing steps was of low availability (e.g. stemming, tagging of part-of-speech, syntax parsing, semantic entity classification, etc.).

In particular, the absence of stemmer is expected to insert incredible noise to the Bag-of-Words based classification. Few stemmers were found in GitHub, but looked quite partial and complex to interface (in particular none was in Python).

Instead, the following simple methods were used for analysis of the data:
Bag of words
Context
word2vec
