import os
    
# Logistic Regression model for Turkish POS tagging using scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from conllu import parse_incr
import numpy as np
import random
import nltk
nltk.download('punkt')
SEED = 1907
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

def sentence_to_tokens(sentence):
	return [token['form'] for token in sentence]

if not os.path.exists("tr_imst-ud-train.conllu"):
	print("datasets downloading...")
	# !wget https://raw.githubusercontent.com/UniversalDependencies/UD_Turkish-IMST/master/tr_imst-ud-train.conllu
	# !wget https://raw.githubusercontent.com/UniversalDependencies/UD_Turkish-IMST/master/tr_imst-ud-test.conllu
	print("downloadin done.")

train_sents = read_conllu("tr_imst-ud-train.conllu")
test_sents = read_conllu("tr_imst-ud-test.conllu")

X_train = [sentence_to_features(s) for s in train_sents]
y_train = [sentence_to_labels(s) for s in train_sents]

X_test = [sentence_to_features(s) for s in test_sents]
y_test = [sentence_to_labels(s) for s in test_sents]

# Flatten the lists for Logistic Regression
X_train_flat = [feat for sent in X_train for feat in sent]
y_train_flat = [label for sent in y_train for label in sent]

X_test_flat = [feat for sent in X_test for feat in sent]
y_test_flat = [label for sent in y_test for label in sent]
# Convert feature dicts to feature vectors
from sklearn.feature_extraction import DictVectorizer
vectorizer = DictVectorizer(sparse=True)
X_train_vec = vectorizer.fit_transform(X_train_flat)
X_test_vec = vectorizer.transform(X_test_flat)

model = LogisticRegression(max_iter=200)
model.fit(X_train_vec, y_train_flat)

y_pred = model.predict(X_test_vec)

print(classification_report(y_test_flat, y_pred))

# Accuracy
accuracy = np.mean(np.array(y_test_flat) == np.array(y_pred))
print(f"Accuracy: {accuracy:.4f}")

# F1-score
from sklearn.metrics import f1_score
f1 = f1_score(y_test_flat, y_pred, average='weighted')
print(f"F1-score: {f1:.4f}")

""" 
         ADJ       0.87      0.59      0.70       960
         ADP       0.89      0.86      0.88       357
         ADV       0.94      0.65      0.77       461
         AUX       0.92      0.67      0.78       211
       CCONJ       0.96      0.90      0.93       356
         DET       0.82      0.97      0.89       344
        INTJ       1.00      0.21      0.35        19
        NOUN       0.61      0.91      0.73      2430
         NUM       0.94      0.53      0.68       192
        PRON       0.95      0.76      0.84       464
       PROPN       0.83      0.30      0.44       374
       PUNCT       0.99      1.00      1.00      1933
        VERB       0.78      0.74      0.76      1928
           X       0.00      0.00      0.00         3
           _       0.88      0.25      0.39       278

    accuracy                           0.79     10310
   macro avg       0.83      0.62      0.68     10310
weighted avg       0.83      0.79      0.79     10310

Accuracy: 0.7938
F1-score: 0.7868
"""