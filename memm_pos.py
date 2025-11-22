# !pip install conllu

import os

if not os.path.exists("tr_imst-ud-train.conllu"):
    print("datasets downloading...")
    # !wget https://raw.githubusercontent.com/UniversalDependencies/UD_Turkish-IMST/master/tr_imst-ud-train.conllu
    # !wget https://raw.githubusercontent.com/UniversalDependencies/UD_Turkish-IMST/master/tr_imst-ud-test.conllu
    print("downloadin done.")
else:
    print("u already have.")

def extract_features(sentence, index, prev_tag):
    token = sentence[index]
    word = token['form']
    
    features = {
        'word': word.lower(),
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        
        'is_capitalized': word[0].isupper(),
        'is_all_caps': word.isupper(),
        'is_all_lower': word.islower(),
        
        'prefix-1': word[0],
        'prefix-2': word[:2],
        'prefix-3': word[:3],
        'suffix-1': word[-1],
        'suffix-2': word[-2:],
        'suffix-3': word[-3:],
        
        'prev_tag': prev_tag,
    }
    return features

from conllu import parse_incr

def prepare_data(filename):
    features = []
    labels = []
    
    with open(filename, "r", encoding="utf-8") as f:
        for sentence in parse_incr(f):
            prev_tag = "<START>"
            
            for i in range(len(sentence)):
                token = sentence[i]
                if token['upos'] is None: continue
                
                features.append(extract_features(sentence, i, prev_tag))
                labels.append(token['upos'])
                
                prev_tag = token['upos']
                
    return features, labels

print("data initilizing...")
train_features, train_labels = prepare_data("tr_imst-ud-train.conllu")
print(f"{len(train_features)} words ready to training.")

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

vectorizer = DictVectorizer(sparse=True)
X_train = vectorizer.fit_transform(train_features)
y_train = train_labels

print("MEMM (MaxEnt) Model training...")
clf = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=100)
clf.fit(X_train, y_train)

print("✅ training done!")

def predict_sentence(sentence_tokens, model, vectorizer):
    predicted_tags = []
    prev_tag = "<START>"
    
    fake_sentence = [{'form': word} for word in sentence_tokens]
    
    for i in range(len(sentence_tokens)):
        features = extract_features(fake_sentence, i, prev_tag)
        
        feature_vector = vectorizer.transform(features)
        
        tag = model.predict(feature_vector)[0]
        predicted_tags.append(tag)
        
        prev_tag = tag
        
    return predicted_tags

sample_sentence = ["Yarın", "okula", "erkenden", "gideceğim", "."]

tags = predict_sentence(sample_sentence, clf, vectorizer)

print("\n--- OUTPUTS ---")
for w, t in zip(sample_sentence, tags):
    print(f"{w} : {t}")