import os
import re
import nltk
import string
import pickle
import neptune
import argparse
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv

load_dotenv()

# Initialize Neptune
project_name = os.getenv('NEPTUNE_PROJECT')
api_token = os.getenv('NEPTUNE_API_TOKEN')

def download_nltk_data():
    nltk.download('punkt')
    nltk.download('stopwords')

def clean_text(text, stop_words, stemmer):
    text = re.sub(r'https?://\S+|www\.\S+|@\S+', '', text)  # Menghapus URL
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)  # Menghapus username
    text = re.sub(r'#[A-Za-z0-9_]+', '', text)  # Menghapus hashtag
    text = text.translate(str.maketrans('', '', string.punctuation))  # Menghapus tanda baca
    text = text.lower()  # Mengubah teks menjadi huruf kecil
    words = word_tokenize(text)  # Tokenisasi teks
    filtered_words = [word for word in words if word not in stop_words]  # Menghapus stopwords
    text = ' '.join([stemmer.stem(word) for word in filtered_words])  # Stemming
    return text

def preprocess_data(input_file, output_dir):

    run = neptune.init_run(project=project_name, api_token=api_token)

    df = pd.read_csv(input_file, encoding='latin1', header=None)
    df = df[[0, 5]]  # Memilih data sentiment dan tweet
    df.columns = ['sentiment', 'tweet']
    df['sentiment'] = df['sentiment'].replace({0: 0, 4: 1})  # Ubah label sentiment dari 0 dan 4 menjadi 0 dan 1

    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    # Menerapkan tahap preprocessing pada kolom 'tweet'
    df['clean_text'] = df['tweet'].apply(lambda x: clean_text(x, stop_words, stemmer))

    vectorizer = TfidfVectorizer(max_features=10000)
    X = vectorizer.fit_transform(df['clean_text'])
    y = df['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    os.makedirs(output_dir, exist_ok=True)

    df.to_csv(os.path.join(output_dir, 'preprocessed_data.csv'), index=False)
    with open(os.path.join(output_dir, 'tfidf_vectorizer.pkl'), 'wb') as file:
        pickle.dump(vectorizer, file)

    train_data = {
        'X_train': X_train,
        'y_train': y_train
    }

    test_data = {
        'X_test': X_test,
        'y_test': y_test
    }

    with open(os.path.join(output_dir, 'train_data.pkl'), 'wb') as file:
        pickle.dump(train_data, file)

    with open(os.path.join(output_dir, 'test_data.pkl'), 'wb') as file:
        pickle.dump(test_data, file)

    # Log preprocessed data and vectorizer to Neptune
    run['data/preprocessed_data'].upload(os.path.join(output_dir, 'preprocessed_data.csv'))
    run['data/vectorizer'].upload(os.path.join(output_dir, 'tfidf_vectorizer.pkl'))
    run['data/train_data'].upload(os.path.join(output_dir, 'train_data.pkl'))
    run['data/test_data'].upload(os.path.join(output_dir, 'test_data.pkl'))

    print("Data preprocessing completed")
    run.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Sentiment140 dataset")
    parser.add_argument('-i', '--input_file', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='Directory to save the processed files')

    args = parser.parse_args()

    download_nltk_data()
    preprocess_data(args.input_file, args.output_dir)
