import pickle
from random import Random

import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import precision_score, recall_score, f1_score


class Models:
    def __init__(self, data_file, word_embedding):
        self.data = data_file
        self.word_embeddings = word_embedding
        self.training_embeddings1 = None
        self.training_embeddings2 = None
        self.training_ambiguous_word_embeddings = None
        self.training_labels = None
        self.testing_embeddings1 = None
        self.testing_embeddings2 = None
        self.testing_ambiguous_word_embeddings = None
        self.testing_labels = None
        self.sentences1_embeddings = None
        self.sentences2_embeddings = None
        self.word_vectors = []
        self.labels = None

    def extract_data(self):
        self.training_embeddings1 = self.data["training_embeddings1"]
        self.training_embeddings2 = self.data["training_embeddings2"]
        self.training_ambiguous_word_embeddings = self.data["train_ambiguous_word_embeddings"]
        self.training_labels = self.data["training_labels"]

        self.testing_embeddings1 = self.data["testing_embeddings1"]
        self.testing_embeddings2 = self.data["testing_embeddings2"]
        self.testing_ambiguous_word_embeddings = self.data["test_ambiguous_word_embeddings"]
        self.testing_labels = self.data["testing_labels"]

        self.sentences1_embeddings = self.data["sentences_embeddings1"]
        self.sentences2_embeddings = self.data["sentences_embeddings2"]
        ambiguous_words = self.data["ambiguous_word"]
        for word in ambiguous_words:
            self.word_vectors.append(self.word_embeddings[word])

    def create_data(self):
        # Create the training_data
        x_train = np.hstack(
            (self.training_ambiguous_word_embeddings, self.training_embeddings1, self.training_embeddings2))
        y_train = self.training_labels

        x_test = np.hstack((self.testing_ambiguous_word_embeddings, self.testing_embeddings1, self.testing_embeddings2))
        y_test = self.testing_labels
        return x_train, y_train, x_test, y_test

    def models(self):
        x_train, y_train, x_test, y_test = self.create_data()
        rfc = RandomForestClassifier()
        # print("Training started..")
        rfc.fit(x_train, y_train)
        # print("Training completed..")
        print("\n")
        print("Random Forest Classifier")
        print("\tTraining:")
        print("\t\t Accuracy score = {:.2f}%".format(rfc.score(x_train, y_train) * 100))
        print("\t\t Precision score = {:.2f}%".format(precision_score(y_train, rfc.predict(x_train)) * 100))
        print("\t\t Recall score = {:.2f}%".format(recall_score(y_train, rfc.predict(x_train)) * 100))
        print("\nTesting:")
        print("\t\t Accuracy score = {:.2f}%".format(rfc.score(x_test, y_test) * 100))
        print("\t\t Precision score = {:.2f}%".format(precision_score(y_test, rfc.predict(x_test)) * 100))
        print("\t\t Recall score = {:.2f}%".format(recall_score(y_test, rfc.predict(x_test)) * 100))
        print("\t\t F1 Score = {:.2f}%".format(f1_score(y_test, rfc.predict(x_test)) * 100))
        filename = "RandomForestClassifier_all-roberta-large-v1"
        joblib.dump(rfc, "Data/"+filename)

        etc = ExtraTreesClassifier()
        # print("Training started..")
        etc.fit(x_train, y_train)
        # print("Training completed..")
        print("\n")
        print("Extra Trees Classifier")
        print("\tTraining:")
        print("\t\t Accuracy score = {:.2f}%".format(etc.score(x_train, y_train) * 100))
        print("\t\t Precision score = {:.2f}%".format(precision_score(y_train, etc.predict(x_train)) * 100))
        print("\t\t Recall score = {:.2f}%".format(recall_score(y_train, etc.predict(x_train)) * 100))
        print("\nTesting:")
        print("\t\t Accuracy score = {:.2f}%".format(etc.score(x_test, y_test) * 100))
        print("\t\t Precision score = {:.2f}%".format(precision_score(y_test, etc.predict(x_test)) * 100))
        print("\t\t Recall score = {:.2f}%".format(recall_score(y_test, etc.predict(x_test)) * 100))
        print("\t\t F1 Score = {:.2f}%".format(f1_score(y_test, etc.predict(x_test)) * 100))
        filename = "ExtraTreesClassifier_all-roberta-large-v1"
        joblib.dump(etc, "Data/"+filename)

        svc = SVC()
        # print("Training started..")
        svc.fit(x_train, y_train)
        # print("Training completed..")
        print("\n")
        print("Support Vector Classifier")
        print("\tTraining:")
        print("\t\t Accuracy score = {:.2f}%".format(svc.score(x_train, y_train) * 100))
        print("\t\t Precision score = {:.2f}%".format(precision_score(y_train, svc.predict(x_train)) * 100))
        print("\t\t Recall score = {:.2f}%".format(recall_score(y_train, svc.predict(x_train)) * 100))
        print("\nTesting:")
        print("\t\t Accuracy score = {:.2f}%".format(svc.score(x_test, y_test) * 100))
        print("\t\t Precision score = {:.2f}%".format(precision_score(y_test, svc.predict(x_test)) * 100))
        print("\t\t Recall score = {:.2f}%".format(recall_score(y_test, svc.predict(x_test)) * 100))
        print("\t\t F1 Score = {:.2f}%".format(f1_score(y_test, svc.predict(x_test)) * 100))
        filename = "SupportVectorClassifier_all-roberta-large-v1"
        joblib.dump(svc, "Data/"+filename)


if __name__ == "__main__":
    word_embeddings = {}
    with open("Glove/glove.6B.50d.txt", 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], 'float32')
            word_embeddings[word] = vector
    with open("Data\Data_all-roberta-large-v1.pkl", "rb") as file:
        data = pickle.load(file)
    ML = Models(data_file=data, word_embedding=word_embeddings)
    ML.extract_data()
    ML.models()
