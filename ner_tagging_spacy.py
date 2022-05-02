import sys, os, glob, json
import numpy as np
import spacy
import time

nlp = spacy.load('en_core_web_sm')

data_path = 'logiqa_data/'
files = ['Eval.txt', 'Test.txt', 'Train.txt']


def ner(text):
    doc = nlp(text)
    ents = []
    for e in doc.ents:
    	ents += e.text.split()
    return list(set(ents))


for filename in files:
    with open(data_path+filename, 'r') as f:
    	lines = f.readlines()
    assert len(lines)%8==0
    n_examples = len(lines) // 8
    out = []
    s = time.time()
    for i in range(n_examples):
    	label_str = lines[i*8+1]
    	context = lines[i*8+2].strip()
    	question = lines[i*8+3]
    	answers = lines[i*8+4 : i*8+8]
    	answers = [a[2:] for a in answers]
    	out.append({'entities': ner("\n".join([context, question] + answers))})
#    	print(out[-1])
    print("writing to {}/{}_ner{}".format(data_path, filename[:-4], filename[-4:]))
    print((time.time()-s))
    with open(data_path + '/' + filename[:-4]+'_ner'+filename[-4:], 'w') as f:
    	json.dump(out, f, indent=4)
#    	print(ner_context)
#    	break
