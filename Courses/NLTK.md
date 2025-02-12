# **Complete Guide to NLTK for Data Scientists** 

Natural Language Toolkit (NLTK) is a powerful Python library for processing human language. This guide covers essential NLTK components, exercises to practice, and a **mini NLP project** at the end. 

---

## **1. Introduction to NLTK** 
NLTK provides tools for tokenization, stemming, lemmatization, part-of-speech tagging, named entity recognition, and more.

### **Installation** 
```bash
pip install nltk
```
To download additional resources: 
```python
import nltk
nltk.download('all')
```

---

## **2. Tokenization** 
Tokenization splits text into words or sentences.

### **Example: Word Tokenization** 
```python
from nltk.tokenize import word_tokenize

sentence = "Natural Language Processing is fun!"
tokens = word_tokenize(sentence)
print(tokens) # Output: ['Natural', 'Language', 'Processing', 'is', 'fun', '!']
```

### **Example: Sentence Tokenization** 
```python
from nltk.tokenize import sent_tokenize

text = "I love NLP. It is an amazing field!"
sentences = sent_tokenize(text)
print(sentences) # Output: ['I love NLP.', 'It is an amazing field!']
```

### **Exercise 1: Tokenization**
- Tokenize `"NLTK makes text processing easy and fun!"` into words and sentences.

---

## **3. Stopword Removal** 
Stopwords are common words (e.g., *the, is, in*) that are often removed in NLP.

### **Example: Removing Stopwords** 
```python
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
words = word_tokenize("This is an amazing NLP tutorial!")
filtered_words = [word for word in words if word.lower() not in stop_words]

print(filtered_words) # Output: ['amazing', 'NLP', 'tutorial', '!']
```

### **Exercise 2: Stopword Filtering**
- Remove stopwords from `"Data Science and Machine Learning are closely related."`

---

## **4. Stemming and Lemmatization** 
These techniques reduce words to their base form.

### **Stemming (Porter Stemmer)** 
```python
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
print(stemmer.stem("running")) # Output: run
print(stemmer.stem("happiness")) # Output: happi
```

### **Lemmatization (WordNet Lemmatizer)** 
```python
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("running", pos="v")) # Output: run
print(lemmatizer.lemmatize("better", pos="a")) # Output: good
```

### **Exercise 3: Stemming and Lemmatization**
- Apply both techniques to `"The cats are running happily in the garden."`

---

## **5. Part-of-Speech (POS) Tagging** 
POS tagging assigns grammatical labels (noun, verb, adjective) to words.

### **Example: POS Tagging** 
```python
from nltk import pos_tag

tokens = word_tokenize("The quick brown fox jumps over the lazy dog")
pos_tags = pos_tag(tokens)

print(pos_tags) # Output: [('The', 'DT'), ('quick', 'JJ'), ('brown', 'JJ'), ('fox', 'NN'), ...]
```

### **Exercise 4: POS Tagging**
- Apply POS tagging to `"Data science involves machine learning and AI."`

---

## **6. Named Entity Recognition (NER)** 
NER identifies names, places, dates, etc.

### **Example: Named Entity Recognition** 
```python
from nltk.chunk import ne_chunk
from nltk import pos_tag

sentence = "Barack Obama was the 44th President of the United States."
tokens = word_tokenize(sentence)
pos_tags = pos_tag(tokens)
named_entities = ne_chunk(pos_tags)

print(named_entities)
```

### **Exercise 5: Named Entity Recognition**
- Extract named entities from `"Elon Musk founded SpaceX in 2002."`

---

## **7. N-grams** 
N-grams capture sequences of words (bigrams, trigrams).

### **Example: Generating Bigrams** 
```python
from nltk.util import ngrams

sentence = "I love data science"
tokens = word_tokenize(sentence)
bigrams = list(ngrams(tokens, 2))

print(bigrams) # Output: [('I', 'love'), ('love', 'data'), ('data', 'science')]
```

### **Exercise 6: Generate Trigrams**
- Create trigrams from `"Machine learning powers AI applications."`

---

## **8. Word Frequency Analysis** 
Finds the most common words in text.

### **Example: Word Frequency** 
```python
from nltk.probability import FreqDist

text = "NLP is amazing. NLP helps machines understand human language."
tokens = word_tokenize(text)
fdist = FreqDist(tokens)

print(fdist.most_common(3)) # Output: [('NLP', 2), ('.', 2), ('is', 1)]
```

### **Exercise 7: Find the Most Common Words**
- Find the top 5 words in `"Deep learning is a subset of machine learning."`

---

# **Mini NLP Project: Sentiment Analysis on Movie Reviews** 

## **Project Goal** 
Build a **sentiment analysis model** using NLTK to classify movie reviews as **positive** or **negative**.

---

## **Step 1: Load and Explore the Dataset** 
- Use `nltk.corpus.movie_reviews` to load movie reviews. 
- Print the number of positive and negative reviews. 
- Check the structure of the dataset (words, labels). 

**Indications:** 
- Use `movie_reviews.fileids('pos')` and `movie_reviews.fileids('neg')`. 
- Read words using `movie_reviews.words(fileid)`. 

---

## **Step 2: Data Preprocessing** 
- Extract words from reviews and clean them. 
- Remove stopwords and punctuation. 

**Indications:** 
- Use `word_tokenize()` for tokenization. 
- Filter out stopwords with `nltk.corpus.stopwords.words('english')`. 
- Convert words to lowercase. 

---

## **Step 3: Feature Extraction** 
- Convert each review into a set of features (word presence). 
- Create a dictionary where words are keys and values are `True` if they appear. 

**Indications:** 
- Use `{word: True for word in words}` to create feature dictionaries. 
- Ensure words are preprocessed before feature extraction. 

---

## **Step 4: Prepare Training and Testing Data** 
- Label positive and negative reviews. 
- Split data into training (80%) and testing (20%). 

**Indications:** 
- Create a list of tuples: `[(features, label)]`. 
- Shuffle data using `random.shuffle()`. 
- Use slicing to split data (`train_data = dataset[:1500]`). 

---

## **Step 5: Train a Naïve Bayes Classifier** 
- Use NLTK’s `NaiveBayesClassifier` to train the model. 
- Print accuracy on the test set. 

**Indications:** 
- Use `NaiveBayesClassifier.train(train_data)`. 
- Evaluate with `accuracy(classifier, test_data)`. 

---

## **Step 6: Test on Custom Reviews** 
- Write a function to classify new reviews. 
- Preprocess input text and extract features. 
- Print whether the review is **positive** or **negative**. 

**Indications:** 
- Use `word_tokenize()` and feature extraction from Step 3. 
- Call `classifier.classify(features)`. 

---

## **Bonus: Improve the Model** 
- Try different feature extraction techniques (bigram presence, TF-IDF). 
- Use a different classifier (e.g., `SklearnClassifier` with SVM or Logistic Regression). 

---

## **Expected Outcome** 
- A model that can classify new movie reviews with reasonable accuracy (~70-80%). 
- Understanding of **text preprocessing, feature extraction, and classification in NLP**. 
