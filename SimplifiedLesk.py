import re

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


class CleanText:
    def __init__(self, sentences):
        self.sentence = sentences

    def sentence_tokenization(self):
        tokenized_sentence = nltk.sent_tokenize(self.sentence.lower())
        print(f"The tokenized sentence is: {tokenized_sentence}.\n")
        return tokenized_sentence

    def remove_punctuations(self, sentence):
        texts = []
        pattern = r"[^a-zA-z0-9\s]"
        for words in sentence:
            text = re.sub(pattern, '', words)
            print(f"The sentence after removing punctuations is: {text}\n")
            texts.append(text)
        return texts

    def stopwords_removal(self, text):
        stop_words = set(stopwords.words("english"))
        context_tab = []
        for sentence in text:
            words = nltk.word_tokenize(sentence)
            without_stop_words = [word for word in words if not word in stop_words]
            context_tab.append(without_stop_words)
        print(f"After stopwords removal {context_tab}.\n")
        return context_tab

    def lemmatization(self, context_tab):
        lemma = []
        wl = WordNetLemmatizer()
        for x in context_tab:
            m2 = []
            for x2 in x:
                x3 = wl.lemmatize(x2, wordnet.VERB)
                x3 = wl.lemmatize(x3, wordnet.NOUN)
                x3 = wl.lemmatize(x3, wordnet.ADJ)
                x3 = wl.lemmatize(x3, wordnet.ADV)
                m2.append(x3)
            lemma.append(m2)
        print(f"The lemmatized words are: {lemma}.\n")
        return lemma

    def parts_of_speech_tagging(self, words):
        pos = []
        for n in words:
            pos.append(nltk.pos_tag(n))
        print("POS (Parts-of-Speech) Tagging")
        for pos_list in pos:
            for word, tag in pos_list:
                print(f"\tWord = {word}, POS Tag = {tag}.")
        return pos

    def driver(self):
        tokenized_sent = self.sentence_tokenization()
        cleaned_text = self.remove_punctuations(tokenized_sent)
        context_words = self.stopwords_removal(cleaned_text)
        lemmatized_words = self.lemmatization(context_words)
        self.parts_of_speech_tagging(lemmatized_words)
        return lemmatized_words


class SimplifiedLesk:
    def __init__(self, text, word):
        self.sentence = text
        self.word = word
        self.stopwords = set(stopwords.words("english"))

    def tokenized_gloss(self, sense):
        tokens = set(word_tokenize(sense.definition()))
        for example in sense.examples():
            tokens.union(set(word_tokenize(example)))
        return tokens

    def compute_overlap(self, signature, context):
        gloss = signature.difference(self.stopwords)
        return len(gloss.intersection(context))

    def disambiguate(self, word, sentence):
        word_senses = wordnet.synsets(word)
        best_sense = word_senses[0]  # Assume that first sense is most freq.
        max_overlap = 0
        context = set(word_tokenize(sentence))
        for sense in word_senses:
            signature = self.tokenized_gloss(sense)
            overlap = self.compute_overlap(signature, context)
            if overlap > max_overlap:
                max_overlap = overlap
                best_sense = sense
        return best_sense


if __name__ == "__main__":
    sentences = str(input("Enter a sentence:"))
    word = str(input("Enter the ambigous word:"))

    # Text-preprocessing
    preprocess = CleanText(sentences=sentences)
    lemmatized_words = preprocess.driver()
    words_list = []
    for text in lemmatized_words:
        for words in text:
            words_list.append(words)
    print(words_list)
    # Lesk Algorithm
    print()
    sentences = nltk.sent_tokenize(sentences.lower())[0]
    lesk = SimplifiedLesk(words_list, word)
    for i in range(0, len(words_list)):
        print("Word :", words_list[i])
        print("Best sense: ", lesk.disambiguate(words_list[i], sentences).definition())
        print()
