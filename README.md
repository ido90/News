# Scraping and Analysis of Hebrew Newspapers
# Abstract

TODO


## Contents

<!--ts-->

[Data](#data)

[Politicians and Parties Appearance Analysis](#politicians-and-parties-appearance-analysis)

[Bag-of-Words Classification](#bag-of-words-classification)

[Context-based Embedding](#context-based-embedding)

<!--te-->

_________________________________

# Data
'Scrapper' package crawls 3 Israeli news websites (**Haaretz, Ynet, Mako**) through their various sections, finds articles and scraps **title, subtitle, text, date and author**.

The crawling found **few hundred articles per website** in March 2019.

Most of the scraped data look valid, with blocked articles in Haaretz being the primary exception.
For more anomalies and some basic properties of the data, see 'Output/Data description and validation'.

![](https://github.com/ido90/News/blob/master/Output/Data%20description%20and%20validation/basic_data_description.png)

## Scope and Limitations
Due to the choice to work with Hebrew text, many of the state-of-the-art of NLP conventional processing steps was of low availability (e.g. stemming, tagging of part-of-speech, syntax parsing, semantic entity classification, etc.).

In particular, the absence of a stemmer has probably inserted incredible noise to the Bag-of-Words classification. Few stemmers were found (e.g. [here](https://github.com/iddoberger/awesome-hebrew-nlp)), but looked quite partial and complex to interface (in particular, none was implemented in Python).

_________________________________

# Politicians and Parties Appearance Analysis

## Goal

Estimate the frequency of the appearance of various political parties and leaders in the media.

## Methodology

Appearances of names of 14 political parties and their leaders were counted in the texts of the articles.

The parties were chosen and sorted according to their success in recent electoral surveys.
Only one leader was counted for each party, even if multiple popular leaders existed (e.g. Gantz and Lapid).

The count of each name includes appearances of the name at the end of words, since a Hebrew name may have an external prefix (e.g. "and") but not an external suffix.

Two different counters were used: one for the total number of appearances of a name, and one for the number of articles in which the name appeared (i.e. counted only once per article).

Note that the appearances may be associated with positive, negative or neutral context, thus the term "popular" below means "they like to write about it" rather than "they like it". 

### Ambiguity

Some of the names are very ambiguous (e.g. Avoda=work, Culanu=all of us, and Levy which is a very common surname). The effect of the ambiguity was attempted to be minimized (e.g. requiring Orly Levy rather than merely Levy, and actually replacing Peretz with Smotrich), yet there may be significant inaccuracies in many names.

**After the corrections, the main possible *overshooting* counters are expected to be Zehut, Culanu and Gesher**.

## Results

- **Netanyahu is clearly the most popular politician in the data**, and appears almost 3 times as the runner-up.
- The gap is smaller when counting articles rather than total appearances. This indicates that the high frequency consists of both many articles including Netanyahu, and dedicated articles which state his name repeatedly.
- Gantz, Haavoda, Zehut and Culanu are very popular as well.
- **Religious and Arab parties are quite under-represented** in the scrapped websites.
- Halikud, Cahol-lavan and Hayamin Hahadash are mostly represented by their leaders, whereas Haavoda and Meretz are mostly represented by the name of the party.
- The representation of the various parties seems quite similar over the 3 websites.
- More insights may be observed once either additional data are scraped, or the count of the names manages to clean the noise of disambiguities.

![](https://github.com/ido90/News/blob/master/Output/Semantic%20stats/Politics%20by%20source.png)

_________________________________

# Bag-of-Words Classification

## Problem definition
In the absence of advanced NLP tools, the old and simple Bag-of-Words concept was used for classification.

The classified units were either full articles or stand-alone paragraphs (with at least 12 words).
The classes were either the source of the text (e.g. ynet) or its section (e.g. economics).

## Methodology
The data were traditionally split into train group (80%) and test group (20%).

To make sure that the test could not be manipulated by simple classification bias, the classes were forced to be equal by omitting data.
This could be avoided by balancing the classifiers through classes weights, and measuring the test results using more complicated metric than accuracy.
However, when the results were displayed vs. the size of the train group, it seemed that the amount of omitted data was smaller than the sensitivity of the classifiers.

### Features
To limit the vocabulary (and also reduce some of the noise), only **Hebrew words** with at least 20 appearances in the (training) data were counted, with a few hundred stopwords filtered out.

Several heuristical features were added to the vocabulary-features: **total words count, average word length, percent of Hebrew characters, percent of English characters and percent of numeric characters**.

To simplify the implementation, the fatures were **not normalized**, but rather left as raw counters and heuristics.
Note that (1) most of the classifiers below should be insensitive to scale, and (2) the scale of the features seemed to be quite simliar in terms of orders of magnitude.

### Classifiers
The following classifiers were used through Scikit-learn:

- **Perceptron**.
- **Random Forest** of 50 entropy-based trees of depth<=4.
- **Naive Bayes** with Laplace (or Lidstone) smoothing of 2 fictive counts per word.
- **SVM** with RBF kernel. Note that this classifier is particularly sensitive to the scale of the features.
- **Neural Network** with a single hidden layer.

No explicit regularization was used (e.g. Lasso or feature-selection), even though it may significantly help to deal with the large number of features along with the not-so-large number of articles.

## Results
### Did we solve the problem?
- Classification to the right **section** among 4 possibilities (news, economics, sports and food) was quite successful for **articles (~90% accuracy)** and reasonable for **paragraphs (~75% accuracy)**. In particular, it means that the **semantics of a field appear in most of the paragraphs of a typical relevant article**, and not only in selected "key" paragraphs.
- According to the **confusion matrix** of the Naive Bayes classifier, news and economics are the most confusing sections.
- Classification of paragraphs to the right **source** among 3 possibilities was less successful, with only **~60% accuracy**. This is intuitively consistent with the previous results of frequency of appearances of politicians in articles, where no significant differences were found between the sources.

### Classifiers comparison
- **Naive Bayes** and the **Neural Network** did well in classification of both articles and paragraphs. **Perceptron** also did well on articles and **SVM** also did well on paragraphs. **Random Forest** performed quite bad in all the tests. Note that quite small efforts were put into tuning of the classifiers, which might have significantly disturbed the results.
- Naive Bayes classifier achieved **impressive results (lower accuracy by few percents) with very small amounts of training data - 50 articles or 300 paragraphs**, which were ~20% of the availavle data.

| ![](https://github.com/ido90/News/blob/master/Output/Classification/Classification%20results%20only%20semantics.png) |
| :--: |
| Classification accuracy using only Bag-of-Words features (without heuristical features) |

### Important features
- In classification of articles, **the various classifiers chose quite different words as dominant factors**, which may indicate that combination of the classifiers may exploit more information and achieve better results. Such a combination was not tried.
- In classification of paragraphs, on the other hand, the classifiers did use quite consistent significant words, such as game, points, vegetables, etc.
- The more successful classifiers **did not assign large importance to the heuristical features** (e.g. average length of words), but rather focused on the Bag-of-Words count. Furthermore, the SVM and NN classifiers seemed to be significantly negatively affected by the existence of those features (which might have been expected in the case of the scale-sensitive SVM).

| ![](https://github.com/ido90/News/blob/master/Output/Classification/features%20paragraph%20to%20section.png) |
| :--: |
| Importance of features in classification of paragraphs to sections (including heuristical features) |

_________________________________

# Context-based Embedding

TODO graph representation, spectral embedding, conductance?

## Word2vec
Word2vec is a vectoric embedding of words, generated such that words with similar context are intended to be close to each other.
Its implementation conceptually [resembles auto-encoders](https://www.tensorflow.org/tutorials/representation/word2vec), though the compressing network aims to reconstruct the word's context (usually defined by skip-grams) rather than the word itself.

Word2vec embedding often has very nice geometric properties. For example, the difference between the embeddings of a country and its capital city is around a constant vector, so in particular one should have 'Berlin'-'Germany'+'Israel'='Tel-Aviv'.

Unfortunately, it turned out to be quite challenging to train a word2vec model for the data of Hebrew articles. While it is possible that the training process was poorly-tuned, it seems more likely that ~18.5K Hebrew sentences with their un-stemmed, highly-variating Hebrew-words suffixes, are just not enough data (e.g. compared to the hundreds of thousands sentences [here](https://www.kaggle.com/c/word2vec-nlp-tutorial)).

The close word2vec-neighborhood of a few selected words in the data is shown below using [t-SNE](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding) embedding.

![](https://github.com/ido90/News/blob/master/Output/Context%20based%20embedding/word2vec_sample_of_neighborhoods.png)
