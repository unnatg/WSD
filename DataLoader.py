import pickle
import re

import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split


class DataLoading:
    def __init__(self, path, word_embeddings):
        self.path = path
        self.word_embeddings = word_embeddings
        self.df = None
        self.X = None
        self.y = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.embeddings1 = None
        self.embeddings2 = None
        self.training_embeddings1 = None
        self.training_embeddings2 = None
        self.testing_embeddings1 = None
        self.testing_embeddings2 = None
        self.word_vectors = []
        self.training_ambiguous_word_embeddings = []
        self.testing_ambiguous_word_embeddings = []

    def load_data(self):
        self.df = pd.read_csv(self.path)
        # Remove rows which have words not present in the word embeddings
        unknown_words = []
        for word in self.df.Ambigous_word.values:
            try:
                _ = word_embeddings[word]
            except KeyError:
                unknown_words.append(word)
        for idx in range(len(self.df)):
            if self.df["Ambigous_word"][idx] in unknown_words:
                self.df.drop(idx, axis=0, inplace=True)

        # Change T/F in the labels column to 1/0
        self.df["Label"] = self.df["Label"].map({"T": 1,
                                                 "F": 0})

    def convert_to_lowercase(self):
        self.df["Sentence1"] = self.df["Sentence1"].str.lower()
        self.df["Sentence2"] = self.df["Sentence2"].str.lower()

    def lemmatize_sentences(self):
        wnl = WordNetLemmatizer()
        sent = []
        for sentences in self.df["Sentence1"]:
            lemma_list = []
            sentences = sentences.split(" ")
            for words in sentences:
                if words != "":
                    lemma_list.append(wnl.lemmatize(words))
            sent.append(" ".join(str(e) for e in lemma_list))
        self.df["Lemmatized_Sentence1"] = sent

        sent = []
        for sentences in self.df["Sentence2"]:
            lemma_list = []
            sentences = sentences.split(" ")
            for words in sentences:
                if words != "":
                    lemma_list.append(wnl.lemmatize(words))
            sent.append(" ".join(str(e) for e in lemma_list))
        self.df["Lemmatized_Sentence2"] = sent

    def remove_special_characters(self, text, remove_digits=True):
        pattern = r"[^a-zA-z0-9\s]"
        cleaned_text = re.sub(pattern, '', text)
        return cleaned_text

    def create_data(self):
        # Remove all special characters
        self.df["Sentence1"] = self.df["Sentence1"].apply(self.remove_special_characters)
        self.df["Sentence2"] = self.df["Sentence2"].apply(self.remove_special_characters)

        self.df["Lemmatized_Sentence1"] = self.df["Lemmatized_Sentence1"].apply(self.remove_special_characters)
        self.df["Lemmatized_Sentence2"] = self.df["Lemmatized_Sentence2"].apply(self.remove_special_characters)

        self.X = self.df.drop(["POS_Tag", "Ambigous_word_index", "Sentence1", "Sentence2", "Label"], axis=1)
        self.y = self.df.Label.values

    def split_data(self):
        # Split data into training and testing
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3,
                                                                                random_state=42)
        print("Training data shape:")
        print(f"\t x_train.shape {self.x_train.shape}, y_train.shape {self.y_train.shape}")
        print("Testing data shape:")
        print(f"\t x_test.shape {self.x_test.shape}, y_test.shape {self.y_test.shape}")

    def get_sentence_embeddings(self):
        model = SentenceTransformer("all-roberta-large-v1")
        # Save the sentences as a list
        sentences1 = self.df["Sentence1"].values
        sentences2 = self.df["Sentence2"].values
        training_sentences1 = self.x_train["Lemmatized_Sentence1"].values
        training_sentences2 = self.x_train["Lemmatized_Sentence2"].values
        testing_sentences1 = self.x_test["Lemmatized_Sentence1"].values
        testing_sentences2 = self.x_test["Lemmatized_Sentence2"].values

        # Get their embeddings
        self.embeddings1 = model.encode(sentences1, show_progress_bar=True)
        self.embeddings2 = model.encode(sentences2, show_progress_bar=True)
        self.training_embeddings1 = model.encode(training_sentences1, show_progress_bar=True)
        self.training_embeddings2 = model.encode(training_sentences2, show_progress_bar=True)
        self.testing_embeddings1 = model.encode(testing_sentences1, show_progress_bar=True)
        self.testing_embeddings2 = model.encode(testing_sentences2, show_progress_bar=True)

    def get_word_embeddings(self):
        for word in self.df.Ambigous_word:
            self.word_vectors.append(self.word_embeddings[word])

        for word in self.x_train.Ambigous_word:
            self.training_ambiguous_word_embeddings.append(self.word_embeddings[word])

        for word in self.x_test.Ambigous_word:
            self.testing_ambiguous_word_embeddings.append(self.word_embeddings[word])

    def save_data(self):
        # All the sentences
        sentences1 = self.df["Sentence1"].values
        sentences2 = self.df["Sentence2"].values
        ambiguous_word = self.df["Ambigous_word"].values
        labels = self.df.Label.values

        # Training sentences
        training_sentences1 = self.x_train["Lemmatized_Sentence1"].values
        training_sentences2 = self.x_train["Lemmatized_Sentence2"].values
        training_ambiguous_words = self.x_train["Ambigous_word"].values

        # Testing sentences
        testing_sentences1 = self.x_test["Lemmatized_Sentence1"].values
        testing_sentences2 = self.x_test["Lemmatized_Sentence2"].values
        testing_ambiguous_words = self.x_test["Ambigous_word"].values

        data = {
            # Training Data
            "training_sentences1": training_sentences1,
            "training_embeddings1": self.training_embeddings1,
            "training_sentences2": training_sentences2,
            "training_embeddings2": self.training_embeddings2,
            "train_ambiguous_word": training_ambiguous_words,
            "train_ambiguous_word_embeddings": self.training_ambiguous_word_embeddings,
            "training_labels": self.y_train,
            # Testing Data
            "testing_sentences1": testing_sentences1,
            "testing_embeddings1": self.testing_embeddings1,
            "testing_sentences2": testing_sentences2,
            "testing_embeddings2": self.testing_embeddings2,
            "test_ambiguous_word": testing_ambiguous_words,
            "test_ambiguous_word_embeddings": self.testing_ambiguous_word_embeddings,
            "testing_labels": self.y_test,
            # Complete Data
            "sentences1": sentences1,
            "sentences_embeddings1": self.embeddings1,
            "sentences2": sentences2,
            "sentences_embeddings2": self.embeddings2,
            "ambiguous_word": ambiguous_word,
            "labels": labels
        }
        # Save the above data to a pickle file
        with open("Data/Data_all-roberta-large-v1.pkl", "wb") as file:
            pickle.dump(data, file)


if __name__ == "__main__":
    csv_path = "Complete.csv"  # File path
    word_embeddings = {}
    with open('Glove/glove.6B.50d.txt', 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], 'float32')
            word_embeddings[word] = vector
    preprocess = DataLoading(csv_path, word_embeddings)
    preprocess.load_data()
    preprocess.convert_to_lowercase()
    preprocess.lemmatize_sentences()
    preprocess.create_data()
    preprocess.split_data()
    preprocess.get_sentence_embeddings()
    preprocess.get_word_embeddings()
    preprocess.save_data()
