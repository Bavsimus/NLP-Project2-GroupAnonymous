import os

if not os.path.exists("tr_imst-ud-train.conllu"):
    print("datasets downloading...")
    # !wget https://raw.githubusercontent.com/UniversalDependencies/UD_Turkish-IMST/master/tr_imst-ud-train.conllu
    # !wget https://raw.githubusercontent.com/UniversalDependencies/UD_Turkish-IMST/master/tr_imst-ud-test.conllu
    print("downloadin done.")
else:
    print("u already have.")
    

# CRF model for Turkish POS tagging using sklearn-crfsuite
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.model_selection import train_test_split
from conllu import parse_incr
import numpy as np
import random
import nltk
nltk.download('punkt')

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)

def read_conllu(file_path):
	with open(file_path, "r", encoding="utf-8") as f:
		data = parse_incr(f)
		sentences = [list(sentence) for sentence in data]
	return sentences

def extract_features(sentence, index):
	word = sentence[index]['form']
	features = {
		'bias': 1.0,
		'word.lower()': word.lower(),
		'word.isupper()': word.isupper(),
		'word.istitle()': word.istitle(),
		'word.isdigit()': word.isdigit(),
	}
	if index > 0:
		prev_word = sentence[index - 1]['form']
		features.update({
			'-1:word.lower()': prev_word.lower(),
			'-1:word.istitle()': prev_word.istitle(),
			'-1:word.isupper()': prev_word.isupper(),
		})
	else:
		features['BOS'] = True  # Beginning of Sentence

	if index < len(sentence) - 1:
		next_word = sentence[index + 1]['form']
		features.update({
			'+1:word.lower()': next_word.lower(),
			'+1:word.istitle()': next_word.istitle(),
			'+1:word.isupper()': next_word.isupper(),
		})
	else:
		features['EOS'] = True  # End of Sentence

	return features

def sentence_to_features(sentence):
	return [extract_features(sentence, i) for i in range(len(sentence))]

def sentence_to_labels(sentence):
	return [token['upos'] for token in sentence]

# Load data
train_sentences = read_conllu("tr_imst-ud-train.conllu")
test_sentences = read_conllu("tr_imst-ud-test.conllu")

# Prepare data
X_train = [sentence_to_features(s) for s in train_sentences]
y_train = [sentence_to_labels(s) for s in train_sentences]

X_test = [sentence_to_features(s) for s in test_sentences]
y_test = [sentence_to_labels(s) for s in test_sentences]

# Train CRF model
crf = sklearn_crfsuite.CRF(
	algorithm='lbfgs',
	c1=0.1,
	c2=0.1,
	max_iterations=100,
	all_possible_transitions=True
)
crf.fit(X_train, y_train)

# Evaluate the model
y_pred = crf.predict(X_test)
print(metrics.flat_classification_report(y_test, y_pred))

# Accuracy
accuracy = metrics.flat_accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# F1-score
f1_score = metrics.flat_f1_score(y_test, y_pred, average='weighted')
print(f"F1-score: {f1_score:.4f}")

# Save the model
""" import joblib
joblib.dump(crf, "turkish_pos_crf_model.pkl")

              precision    recall  f1-score   support

         ADJ       0.88      0.66      0.76       960
         ADP       0.93      0.88      0.91       357
         ADV       0.92      0.72      0.81       461
         AUX       0.93      0.77      0.84       211
       CCONJ       0.97      0.94      0.95       356
         DET       0.85      0.98      0.91       344
        INTJ       0.88      0.37      0.52        19
        NOUN       0.67      0.87      0.76      2430
         NUM       0.94      0.61      0.74       192
        PRON       0.95      0.85      0.90       464
       PROPN       0.83      0.37      0.51       374
       PUNCT       1.00      1.00      1.00      1933
        VERB       0.76      0.80      0.78      1928
           X       0.00      0.00      0.00         3
           _       0.78      0.40      0.53       278

    accuracy                           0.82     10310
   macro avg       0.82      0.68      0.73     10310
weighted avg       0.84      0.82      0.82     10310

Accuracy: 0.8210
F1-score: 0.8171 """