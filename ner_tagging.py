import sys, os, glob, json
import numpy as np
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize

data_path = 'logiqa_data/'
files = ['Eval.txt', 'Test.txt', 'Train.txt']

st = StanfordNERTagger('stanford-ner-2020-11-17/classifiers/english.all.3class.distsim.crf.ser.gz',
			'stanford-ner-2020-11-17/stanford-ner.jar', encoding='utf-8')

def ner(text):
    entities = {}
#    indices = []
    ner_text = st.tag(word_tokenize(text))
    for idx, (t, p) in enumerate(ner_text):
    	if p != 'O':
    	    entities[t] = entities.get(t, []) + [idx]
#    	indices.append(int(p!='O'))
    return entities #, indices


for filename in files:
    with open(data_path+filename, 'r') as f:
    	lines = f.readlines()
    assert len(lines)%8==0
    n_examples = len(lines) // 8
    out = []
    for i in range(n_examples):
    	label_str = lines[i*8+1]
    	context = lines[i*8+2].strip()
    	question = lines[i*8+3]
    	answers = lines[i*8+4 : i*8+8]

    	out.append({'context':ner(context), 'question':ner(question),
    		    'answers':[ner(a) for a in answers]})
        entities = list(out[-1]['context'].keys()) + list(out[-1]['question'].keys()) + [list(a.keys()) for a in out[-1]['answers']]
        out[-1]['entities'] = list(set(entities))
    	print(out[-1])
    print("writing to {}/{}_ner{}".format(data_path, filename[:-4], filename[-4:]))
    with open(data_path + '/' + filename[:-4]+'_ner'+filename[-4:], 'w') as f:
    	json.dump(out, f, indent=4)
#    	print(ner_context)
#    	break
