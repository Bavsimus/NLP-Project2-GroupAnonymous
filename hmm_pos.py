# !pip install conllu

import os

# !wget https://raw.githubusercontent.com/UniversalDependencies/UD_Turkish-IMST/master/tr_imst-ud-train.conllu
# !wget https://raw.githubusercontent.com/UniversalDependencies/UD_Turkish-IMST/master/tr_imst-ud-test.conllu

print("files downloaded: tr_imst-ud-train.conllu, tr_imst-ud-test.conllu")

import numpy as np
from collections import defaultdict
from conllu import parse_incr

class HMMPOSTagger:
    def __init__(self):
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.emission_counts = defaultdict(lambda: defaultdict(int))
        self.tag_counts = defaultdict(int)
        
        self.vocab = set()
        self.tags = set()
        
        self.transition_probs = {}
        self.emission_probs = {}

    def train(self, data_file_path):
        print(f"Training model with data from: {data_file_path}...")
        
        with open(data_file_path, "r", encoding="utf-8") as file:
            for tokenlist in parse_incr(file):
                
                prev_tag = "<START>" 
                
                for token in tokenlist:
                    word = token['form'].lower()
                    tag = token['upos']
                    
                    if tag is None: continue
                    
                    self.transition_counts[prev_tag][tag] += 1
                    
                    self.emission_counts[tag][word] += 1
                    
                    self.tag_counts[tag] += 1
                    self.tag_counts[prev_tag] += 1
                    
                    self.vocab.add(word)
                    self.tags.add(tag)
                    prev_tag = tag
                    
        epsilon = 1e-6
        
        for prev_tag in self.transition_counts:
            total = sum(self.transition_counts[prev_tag].values())
            self.transition_probs[prev_tag] = {}
            for tag in self.tags:
                count = self.transition_counts[prev_tag].get(tag, 0)
                self.transition_probs[prev_tag][tag] = np.log((count + epsilon) / (total + epsilon * len(self.tags)))

        for tag in self.tags:
            total = sum(self.emission_counts[tag].values())
            self.emission_probs[tag] = {}
            for word in self.vocab:
                count = self.emission_counts[tag].get(word, 0)
                self.emission_probs[tag][word] = np.log((count + epsilon) / (total + epsilon * len(self.vocab)))
                
        print("Training complete!")

    def viterbi(self, sentence):
        tokens = [w.lower() for w in sentence]
        T = len(tokens)
        tag_list = list(self.tags)
        N = len(tag_list)
        
        viterbi_matrix = np.full((T, N), -1e9)
        backpointer = np.zeros((T, N), dtype=int)
        
        start_token = tokens[0]
        for idx, tag in enumerate(tag_list):
            trans_p = self.transition_probs.get("<START>", {}).get(tag, -100)
            
            if start_token in self.vocab:
                emit_p = self.emission_probs.get(tag, {}).get(start_token, -100)
            else:
                emit_p = -15
            
            viterbi_matrix[0][idx] = trans_p + emit_p

        for t in range(1, T):
            word = tokens[t]
            for curr_idx, curr_tag in enumerate(tag_list):
                
                if word in self.vocab:
                    emit_p = self.emission_probs.get(curr_tag, {}).get(word, -100)
                else:
                    emit_p = -15
                
                best_prob = -1e9
                best_prev_idx = 0
                
                for prev_idx, prev_tag in enumerate(tag_list):
                    trans_p = self.transition_probs.get(prev_tag, {}).get(curr_tag, -100)
                    prob = viterbi_matrix[t-1][prev_idx] + trans_p + emit_p
                    
                    if prob > best_prob:
                        best_prob = prob
                        best_prev_idx = prev_idx
                
                viterbi_matrix[t][curr_idx] = best_prob
                backpointer[t][curr_idx] = best_prev_idx

        best_path = []
        best_last_idx = np.argmax(viterbi_matrix[T-1])
        best_path.append(best_last_idx)
        
        for t in range(T-1, 0, -1):
            best_last_idx = backpointer[t][best_last_idx]
            best_path.insert(0, best_last_idx)
            
        return [tag_list[i] for i in best_path]
    
hmm = HMMPOSTagger()

hmm.train("tr_imst-ud-train.conllu")

sample_sentence = ["Bugün", "hava", "çok", "güzel", "."]
tags = hmm.viterbi(sample_sentence)

print("\n--- RESULTS ---")
for w, t in zip(sample_sentence, tags):
    print(f"{w} : {t}")