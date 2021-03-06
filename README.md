# WSD

WSD - Word-sense disambiguation is an open problem in Natural Language Processing concerned with identifying which sense of a word is used in a sentence.
The major purpose of WSD technique is to accurately understand the meaning of a specific word in a text. We have proposed a method to find ambiguous words in a sentence employing a supervised machine learning approach. If the word has different meaning in both the sentences then it is domain dependent otherwise domain independent. 


## Step1: Import all the required libraries

## Step2: Generate word embeddings, glove as well as ROBERTA  
		- Lemmatize sentence1 and sentence2  
		- Drop all the irrelevant columns except Ambigous_Word, Sentence1, Sentence2. Store it as X  
		- Convert the labels column into 1 or 0 and store it as y.  
		- Perform train_test_split()  
		- Store the data in the pickle file.  
		- Store the whole data (X, y) in the pickle file.  

### Load the data from the pickle file  
	Pickle file format:
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

## Step3: Train a machine learning model (Random Forest, Extra Trees Classifier and Support Vector Classifier), with the following format:  
		  input_data=<ambigous_word_embeddings, sentence1_embeddings, sentence2_embeddings>  
		  input_data=<GloVe Vector Embeddings,  ROBERTA,              ROBERTA>  

## Step4: Evaluate the model using testing data and development data. Evaluation metrics = accuracy, precision, recall, f1 score  

## Step5: Prediction phase:  
		  - Input two sentences from the user.  
		  - Convert the sentences to lowercase. Remove stopwords  
	          - Lemmatize the words  
	          - Get glove embedding for noun and the ROBERTA embeddings for sentences and predict using the classifier whether T or F.  
		  
