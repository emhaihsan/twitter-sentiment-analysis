# Twitter Sentiment Data Preprocessing
This project demonstrates the process of preprocessing Twitter data for sentiment analysis purposes

## Dataset Overview
The Sentiment 140 dataset contains 1.6 million tweets extracted using the Twitter API. Data is annotated with labels 0 which indicates negative and 4 which indicates positive. This dataset consists of 6 columns:
- target: Sentiment
- ids : Tweet ID
- date: Tweet date
- flag : Query Flag
- user : Username of the tweet author
- text: Tweet text

Source: Go, A., Bhayani, R. and Huang, L., 2009. Twitter sentiment classification using distant supervision. CS224N Project Report, Stanford, 1(2009), p.12. [(Download Paper)](https://www-cs.stanford.edu/people/alecmgo/papers/TwitterDistantSupervision09.pdf)

Download Dataset : [Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)


## Text Preprocessing

The initial step in this project is to carry out text preprocessing from existing data. After the dataset is loaded, several adjustments are made, including:
1. Only select the relevant columns, namely the `text` column as the tweet data to be analyzed and the `target` column which is the sentiment to be predicted.
2. Change the label for positive sentiment from 4 to 1, so that the final predicted label result is 0 for negative and 1 for positive.

The next step is to carry out data cleaning to ensure that the data processed by machine learning is clean and does not contain a lot of noise. The following are the preprocessing stages carried out.
1. **Removing URL**
2. **Deleting Username**
3. **Deleting Hashtags**
4. **Removing Punctuation**
5. **Convert to Lowercase**
6. **Tokenization** : Breaking down text into individual word tokens.
7. **Deleting Stopwords** : Removes common words such as conjunctions (for example, "the", "is", "in").
8. **Stemming**: Changes words into their basic forms.

## Text Vectorizing
At this stage, the process of converting text into a numerical representation is carried out. The numeric form of data will make it easier for machine learning to carry out model processing. For this stage, the TF-IDF method is used which is usually applied to assess how important a word in a document is relative to a collection of documents (corpus). TF-IDF is the result of multiplying Term Frequency (TF) and Inverse Document Frequency (IDF).

**Term Frequency (TF)**

Term Frequency (TF) measures how often a word appears in a document.

$$ \text{TF}(t, d) = \frac{\text{Number of occurrences of word } t \text{ in document } d}{\text{Total number of words in document } d} $$

**Inverse Document Frequency (IDF)**

Inverse Document Frequency (IDF) measures how important a word is in the entire corpus.

$$ \text{IDF}(t, D) = \log \left( \frac{N}{|\{d \in D : t \in d\}|} \right) $$

Where:
- \( N \) is the total number of documents in the corpus.
- $(|\{d \in D : t \in d\}|)$ is the number of documents containing the word \( t \).

**TF-IDF**

TF-IDF is the product of TF and IDF.

$$ \text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D) $$

Where:
- \( t \) is a specific word.
- \( d \) is a specific document in the corpus.
- \( D \) is the entire corpus of documents.


## Text Splitting
To allow a more objective evaluation of the model. Text splitting was carried out by dividing the overall data into 80% for training and 20% testing. The presence of testing data that is not used for training is useful for providing an overview of the model's prediction accuracy that has never been seen before.
