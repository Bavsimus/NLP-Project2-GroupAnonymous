import os

# Rule based model for Turkish POS tagging
# Rules :
# if it is a end word of sentence, tag as PUNCT
# if it begins with a capital letter, tag as PROPN
# if it is all uppercase, tag as PROPN
# if it contains only digits, tag as NUM
# if it ends with "iyor", "ıyor", "uyor", "üyor" tag as VERB
# if it ends with "li", "lı", "lu", "lü", tag as ADJ
# if it ends with "mek", "mak", tag as VERB
# else tag as NOUN

from conllu import parse_incr
import random
import numpy as np

SEED = 1907
random.seed(SEED)
np.random.seed(SEED)
def read_conllu(file_path):
	with open(file_path, "r", encoding="utf-8") as f:
		data = parse_incr(f)
		sentences = [list(sentence) for sentence in data]
	return sentences

train_sents = read_conllu("tr_imst-ud-train.conllu")
test_sents = read_conllu("tr_imst-ud-test.conllu")

def rule_based_tagger(word, is_end_of_sentence):
	if is_end_of_sentence:
		return "PUNCT"
	elif word[0].isupper():
		return "PROPN"
	elif word.isdigit():
		return "NUM"
	elif word.endswith(("iyor", "ıyor", "uyor", "üyor")):
		return "VERB"
	elif word.endswith(("li", "lı", "lu", "lü")):
		return "ADJ"
	elif word.endswith(("mek", "mak")):
		return "VERB"
	else:
		return "NOUN"

def evaluate_rule_based_tagger(sentences):
	correct = 0
	total = 0
	
	for sentence in sentences:
		for i, token in enumerate(sentence):
			word = token['form']
			true_tag = token['upos']
			is_end_of_sentence = (i == len(sentence) - 1)
			predicted_tag = rule_based_tagger(word, is_end_of_sentence)
			
			if predicted_tag == true_tag:
				correct += 1
			total += 1
	
	accuracy = correct / total
	return accuracy

train_accuracy = evaluate_rule_based_tagger(train_sents)
test_accuracy = evaluate_rule_based_tagger(test_sents)

print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

"""
Train Accuracy: 0.3654
Test Accuracy: 0.3563
"""

