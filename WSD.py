import re
import pandas as pd
import joblib
import numpy as np
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


class Prediction:
    def __init__(self, input1, input2, ambiguous_word, word_embedding):
        self.word_embeddings = word_embedding
        self.word = ambiguous_word
        self.sentence1 = input1
        self.sentence2 = input2

    def preprocess_inputs(self):
        # Convert to lowercase
        self.sentence1 = self.sentence1.lower()
        self.sentence2 = self.sentence2.lower()

        # Remove special characters from strings
        sentence1 = re.sub(r"[^a-zA-Z0-9]", " ", self.sentence1)
        sentence2 = re.sub(r"[^a-zA-Z0-9]", " ", self.sentence2)

        # Lemmatize the sentences
        wnl = WordNetLemmatizer()
        lemma_list = []
        for words in sentence1:
            lemma_list.append(wnl.lemmatize(words))
        lemmatized_sentence1 = " ".join(str(e) for e in lemma_list)

        lemma_list = []
        for words in sentence2:
            lemma_list.append(wnl.lemmatize(words))
        lemmatized_sentence2 = " ".join(str(e) for e in lemma_list)

        return lemmatized_sentence1, lemmatized_sentence2

    def get_sentence_embeddings(self):
        sentence1, sentence2 = self.preprocess_inputs()
        model = SentenceTransformer("all-roberta-large-v1")
        sentence1_encoding = model.encode(sentence1, show_progress_bar=False)
        sentence2_encoding = model.encode(sentence2, show_progress_bar=False)
        word_vector = self.word_embeddings[self.word]
        return sentence1_encoding, sentence2_encoding, word_vector
    
    def evaluation(self):
        eval_data = pd.read_csv("Dev.csv")
        eval_data["Label"] = eval_data["Label"].map({"T": 1,
                                                 "F": 0})
        y_test = eval_data["Label"].values
        ambiguous_word = eval_data["Ambigous_word"].values
        word_vectors = []
        for word in ambiguous_word:
            try:
                word_vectors.append(self.word_embeddings[word])
            except KeyError:
                word_vectors.append(np.zeros(50))
        lemmatized_sentence1 = eval_data["Lemmatized_Sentence1"]
        lemmatized_sentence2 = eval_data["Lemmatized_Sentence2"]
        model = SentenceTransformer("all-roberta-large-v1")
        sentence1_encoding = model.encode(lemmatized_sentence1, show_progress_bar=True)
        sentence2_encoding = model.encode(lemmatized_sentence2, show_progress_bar=True)
        test_data = np.hstack((word_vectors, sentence1_encoding, sentence2_encoding))
        rfc = joblib.load("Data/RandomForestClassifier_all-roberta-large-v1")
        etc = joblib.load("Data/ExtraTreesClassifier_all-roberta-large-v1")
        svc = joblib.load("Data/SupportVectorClassifier_all-roberta-large-v1")
        rfc_prediction = rfc.predict(test_data)
        etc_prediction = etc.predict(test_data)
        svc_prediction = svc.predict(test_data)

        # Accuracy results
        print("Results by Random Forest Classifier:")
        print("\t Accuracy score = {:.2f}%".format(accuracy_score(y_test, rfc_prediction) * 100))
        print("\t Precision score = {:.2f}%".format(precision_score(y_test, rfc_prediction) * 100))
        print("\t Recall score = {:.2f}%".format(recall_score(y_test, rfc_prediction) * 100))
        print("\t F1 Score = {:.2f}%".format(f1_score(y_test, rfc_prediction) * 100))

        print("\nResults by Extra Trees Classifier:")
        print("\t Accuracy score = {:.2f}%".format(accuracy_score(y_test, etc_prediction) * 100))
        print("\t Precision score = {:.2f}%".format(precision_score(y_test, etc_prediction) * 100))
        print("\t Recall score = {:.2f}%".format(recall_score(y_test, etc_prediction) * 100))
        print("\t F1 Score = {:.2f}%".format(f1_score(y_test, etc_prediction) * 100))

        print("\nResults by Support Vector Classifier:")
        print("\t Accuracy score = {:.2f}%".format(accuracy_score(y_test, svc_prediction) * 100))
        print("\t Precision score = {:.2f}%".format(precision_score(y_test, svc_prediction) * 100))
        print("\t Recall score = {:.2f}%".format(recall_score(y_test, svc_prediction) * 100))
        print("\t F1 Score = {:.2f}%".format(f1_score(y_test, svc_prediction) * 100))       

    def predict(self):
        # Load the model
        rfc = joblib.load("Data/RandomForestClassifier_all-roberta-large-v1")
        etc = joblib.load("Data/ExtraTreesClassifier_all-roberta-large-v1")
        svc = joblib.load("Data/SupportVectorClassifier_all-roberta-large-v1")
        # Prepare the data
        sentence1, sentence2, word = self.get_sentence_embeddings()
        inputs = np.hstack((word, sentence1, sentence2))
        rfc_prediction = rfc.predict(inputs.reshape(1, -1))
        etc_prediction = etc.predict(inputs.reshape(1, -1))
        svc_prediction = svc.predict(inputs.reshape(1, -1))
        # predictions_list = [rfc_prediction[0], etc_prediction[0], svc_prediction[0]]
        # prediction_one = 0
        # prediction_zero = 0
        # for value in predictions_list:
        #     if value == 1:
        #         prediction_one = prediction_one + 1
        #     else:
        #         prediction_zero = prediction_zero + 1
        # prediction = 1 if prediction_one>prediction_zero else 0
        # print()
        print("\nPrediction by Random Forest Classifier:")
        if rfc_prediction[0] == 1:
            print(f"The word '{self.word}' is ambiguous. It has a different meaning in both the sentences.")
        else:
            print(
                f"The word '{self.word}' is not ambiguous, it has the same meaning in both the sentences."
                f' Thus, it is "domain independent".')
        
        print("\nPrediction by Extra Trees Classifier:")
        if etc_prediction[0] == 1:
            print(f"The word '{self.word}' is ambiguous. It has a different meaning in both the sentences.")
        else:
            print(
                f"The word '{self.word}' is not ambiguous, it has the same meaning in both the sentences."
                f' Thus, it is "domain independent".')

        print("\nPrediction by Support Vector Classifier:")
        if svc_prediction[0] == 1:
            print(f"The word '{self.word}' is ambiguous. It has a different meaning in both the sentences.")
        else:
            print(
                f"The word '{self.word}' is not ambiguous, it has the same meaning in both the sentences."
                f' Thus, it is "domain independent".')


if __name__ == "__main__":
    sentence1 = "He walked into a coal mine."
    sentence2 = "He stepped on a land mine."
    ambiguous_word = "mine"
    word_embeddings = {}
    with open("Glove/glove.6B.50d.txt", 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], 'float32')
            word_embeddings[word] = vector
    wsd = Prediction(sentence1, sentence2, ambiguous_word, word_embeddings)
    wsd.predict()
    print("\n")
    # wsd.evaluation()
