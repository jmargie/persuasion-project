### runfacts.py
### julia margie
### splits paragraphs into sentences and clauses, and records how many of those 
### clauses are facts and opinions.


#split into sentences
import pandas as pd
from transformers import pipeline
import torch
import spacy

#note: opinion = 0, fact = 1

# edited list of punctuation via SpaCy that indicates end of sentence to include somethat
# ended ideas in the sentences I was looking at
punc = ['!', '.', '?', ':', '؟', '۔', '܀', '܁', '܂', '߹', '।', '॥', '၊', '။', '።',
 '፧', '፨', '᙮', '/', '᜶', '᠃', '᠉', '᥄', '᥅', '᪨', '᪩', '᪪', '᪫',
 '᭚', '᭛', '᭞', '᭟', '᰻', '᰼', '᱾', '᱿', '‼', '‽', '⁇', '⁈', '⁉',
 '⸮', '⸼', '=', '*', '꘏', '꛳', '꛷', '꡶', '꡷', '꣎', '꣏', '꤯', '꧈',
 '꧉', '꩝', '꩞', '꩟', '꫰', '꫱', '꯫', '﹒', '﹖', '﹗', '！', '．', '？',
 '𐩖', '𐩗', '𑁇', '𑁈', '𑂾', '𑂿', '𑃀', '𑃁', '𑅁', '𑅂', '𑅃', '𑇅',
 '𑇆', '𑇍', '𑇞', '𑇟', '𑈸', '𑈹', '𑈻', '𑈼', '𑊩', '𑑋', '𑑌', '𑗂',
 '𑗃', '𑗉', '𑗊', '𑗋', '𑗌', '𑗍', '𑗎', '𑗏', '𑗐', '𑗑', '𑗒', '𑗓',
 '𑗔', '𑗕', '𑗖', '𑗗', '𑙁', '𑙂', '𑜼', '𑜽', '𑜾', '𑩂', '𑩃', '𑪛',
 '𑪜', '𑱁', '𑱂', '𖩮', '𖩯', '𖫵', '𖬷', '𖬸', '𖭄', '𛲟', '𝪈', '｡', '。', ';', ' - ']

#load data, model, and tokenizer
data = pd.read_csv('with_scores.tsv', sep = '\t', header = 0)
model = 'factsmodel'
tokenizer = 'factstokenizer'


english = spacy.load('en_core_web_sm')

#try to put on GPU
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print(x)
else:
    print ("MPS device not found.")



def segment_clauses(text):
#function which takes a paragraph and splits it into clauses, with one idea each
#returns list of clauses

# splits based on relative or auxilary clause, because those are always new ideas, 
# whereas clausal complements and similar, would split clauses into single words, often
# all relations from: https://universaldependencies.org/en/dep/index.html
    spacy.punct_chars = punc
    doc = english(text) #turn into sentences
    clauses = []
    for sentence in doc.sents:
        clause = []
        for token in sentence: 
            if token.dep_ in ('relcl', 'aux') and clause: 	#if word's dependency indicates
                clauses.append(" ".join(clause))			# it starts a clause, add clause
                clause = []									# to list and start clause formation again
                clause.append(token.text.strip())
            else:
                clause.append(token.text.strip())

        if clause:
            clauses.append(" ".join(clause))
    return clauses

#load model onto gpu
classifier = pipeline("text-classification", model = model, tokenizer = tokenizer, device = 'mps:0')

def count_facts(clauses):
    #function which takes a list of clauses and counts how many are facts or opinions
    countf = 0
    counto = 0
    for clause in clauses:
        if clause:
            mark = list(classifier(clause)[0].values())[0]
            if mark == 'LABEL_1':
                countf +=1
            else:
                counto +=1
    return [countf, counto]

#create columns
data['facts'] = 0
data['opinions'] = 0

#split each argument into clauses and make a dataframe pointing to those lists
segmented = pd.DataFrame((data['argument'].apply(lambda x : segment_clauses(x))))

#for efficiency and testing, looped through instead of just using .apply
#this way, I could stop it at any point, and keep track of where in the run I was
#also, only had to run count_facts once, because I could not figure out how to use 
# one .apply to assign two columns
for index in segmented['argument'].index:
    counter = count_facts(segmented['argument'].iloc[index])
    data.loc[index, 'facts'] = counter[0]
    data.loc[index, 'opinions'] = counter[1]
    

#save to computer
data.to_csv("with_facts.tsv", sep='\t', index=False)
            