# -*- coding: utf-8 -*-
"""Adding_Irrelevant_Info.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17uu6KMnvGxdWhkLm_bcTt8s6pD2fojcZ
"""

import pandas as pd
import datasets
from datasets import load_dataset
import nltk
from nltk import sent_tokenize
from nltk.corpus import brown
from nltk.tokenize.moses import MosesDetokenizer
import random
from enum import Enum
import os

# Download and preprocess Shakespearean text
dataset = load_dataset('tiny_shakespeare')['train']
nltk.download('punkt')
shakespearean_text = [text.replace('\n', ' ') for text in sent_tokenize(dataset['text'][0])]

# Download and preprocess Brown corpus
nltk.download('brown')
nltk.download('perluniprops')
mdetok = MosesDetokenizer()
brown_natural = [mdetok.detokenize(' '.join(sent).replace('``', '').replace("''", '').replace('`', "").split(), return_str=True)  for sent in brown.sents()]

# Ensure that Train, Eval and Test files are in the same directory if not downloading
# !wget https://raw.githubusercontent.com/lgw863/LogiQA-dataset/master/Train.txt -O Train.txt
# !wget https://raw.githubusercontent.com/lgw863/LogiQA-dataset/master/Eval.txt -O Eval.txt
# !wget https://raw.githubusercontent.com/lgw863/LogiQA-dataset/master/Test.txt -O Test.txt

class Position(Enum):
  PREPEND = 1
  IN_BETWEEN = 2
  APPEND = 3

def prepare_irrelavent_data(filename, pos: Position, irrelevant_data_source, 
                            out_file_name):
  
  with open(filename, 'r') as f:
    lines = f.readlines()
  
  assert len(lines)%8==0
  n_examples = len(lines) // 8

  out_file_name = filename[:filename.index('.')] + '_' + out_file_name + '.txt'

  if os.path.exists(out_file_name):
    os.remove(out_file_name)

  for i in range(n_examples):
    label_str = lines[i*8+1]
    context = lines[i*8+2].strip()
    question = lines[i*8+3]
    answers = lines[i*8+4 : i*8+8]

    irrelevant_data = random.choice(irrelevant_data_source)
    
    if pos == Position.PREPEND:
      context = irrelevant_data + ' ' + context + '\n'
    elif pos == Position.IN_BETWEEN:
      sentences = sent_tokenize(context)
      sentences.insert(random.randint(0, len(sentences)), irrelevant_data)
      context = ' '.join(sentences) + '\n'
    else:
      context += ' ' + irrelevant_data + '\n'

    with open(out_file_name, 'a') as of:
      of.write('\n')
      of.write(label_str)
      of.write(context)
      of.write(question)
      for i in range(4):
        of.write(answers[i])

data_sources = {'shakespeare': shakespearean_text, 'brown': brown_natural}
logiQA_files = ['Train.txt', 'Eval.txt', 'Test.txt']
position = Position.APPEND
for file in logiQA_files:
  for data_source_name in data_sources:
    prepare_irrelavent_data(file, position, data_sources[data_source_name], data_source_name)