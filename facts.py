### facts.py
### julia margie
### load and train HuggingFace transformers to identify facts vs opinions

#note: opinion = 0, fact = 1
import pandas as pd
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import Trainer, TrainingArguments
from transformers import DebertaForSequenceClassification, DebertaTokenizer, DataCollatorWithPadding


def add_label(example, num):
    #function which adds a 1 or a 0 to the label column of the training dataset
    example['label'] = num
    return example

def preprocess_function(examples):
    # function which tokenizes a sentence  
	# returns an array with those vectors and a fact/opinion label
    tokenized = tokenizer1(examples['text'],
                           padding=True, 
                           truncation=False)
    tokenized['labels'] = examples['label']
    return tokenized



#JUST FACTS dataset and choose random 4000 for less compute time
dataset = load_dataset("fever/fever", 'wiki_pages')
dataset = dataset.shuffle(seed=42)
facts_dataset = dataset['wikipedia_pages'].select(range(4000))
facts_dataset = facts_dataset.map(add_label, 1)



#JUST OPINIONS dataset and choose random
datasetopinions = pd.read_csv("all.txt.data.txt", sep = '\t', header = 0)
datasetop = Dataset.from_pandas(datasetopinions) #turn into a huggingface dataset for consistency
datasetop = datasetop.shuffle(seed=42)
opinions_dataset = datasetop.select(range(4000))
opinions_dataset = opinions_dataset.map(add_label, 0)



#dataset with BOTH labels for 
datasetboth_orig = pd.read_csv("facts_opinions.csv", header = 0)
datasetboth = Dataset.from_pandas(datasetboth_orig)
datasetboth = datasetboth.shuffle(seed=42)
datasetboth = datasetboth.select(range(4000))


#load pretrained model and tokenizer
model_name = "microsoft/deberta-base"  
model = DebertaForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer1 = DebertaTokenizer.from_pretrained("microsoft/deberta-base")
#make all vectors same length
data_collator1 = DataCollatorWithPadding(tokenizer=tokenizer1)


# processing the data
facts_dataset.remove_columns(['id', 'lines'])
datasetTRAIN = concatenate_datasets([facts_dataset, opinions_dataset])


# mapping with function
train_full = datasetTRAIN.map(lambda x: preprocess_function(x), batched=True)
test_full = datasetboth.map(lambda x: preprocess_function(x), batched=True)

training_args = TrainingArguments(
    output_dir="fact_train",
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    gradient_accumulation_steps=4,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_full,
    eval_dataset=test_full,
    tokenizer = tokenizer1,
    data_collator = data_collator1
)


# train the model
trainer.train()
print(trainer.evaluate())

model.save_pretrained("/Users/juliamargie/Documents/GitHub/persuation2/factsmodel")
print("\n \n SAVING COMPLETE \n \n")

# Save the tokenizer
tokenizer1.save_pretrained("/Users/juliamargie/Documents/GitHub/persuation2/factstokenizer")
print("\n \n TOKENIZER COMPLETE \n \n")
