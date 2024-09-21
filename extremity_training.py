###   extremity_training.py
###   julia margie
### Using the below data to train Microsoft's roBERTa to look at overall emotional intensity

'''@InProceedings{MohammadB17starsem,
	Title={Emotion Intensities in Tweets},
	author={Mohammad, Saif M. and Bravo-Marquez, Felipe},
	booktitle={Proceedings of the sixth joint conference on lexical and computational semantics (*Sem)}, 
	address = {Vancouver, Canada},
	year={2017}
}'''


#ORDER IN ALL LISTS: [anger, joy, sadness, fear]

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, pipeline

#data in huggingface, so wanted to keep the model there too, since it was my first time using a transformer
model_name = "cardiffnlp/twitter-roberta-base-sentiment"

# load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# function to add columns to the dataset with classification and score
def preprocess(examples):
    tokenized = tokenizer(examples['sentence'], 
                          padding=True, 
                          truncation=False)
    tokenized["labels"] = examples['intensity']
    return tokenized

datafiles = ['data/anger-ratings-0to1.train.txt', 'data/joy-ratings-0to1.train.txt',
              'data/sadness-ratings-0to1.train.txt','data/fear-ratings-0to1.train.txt',
              'data/anger-ratings-0to1.test.txt', 'data/joy-ratings-0to1.test.txt',
              'data/sadness-ratings-0to1.test.txt','data/fear-ratings-0to1.test.txt',]

train_data = [None] * 4
test_data = [None] * 4

# load datasets
for i in range(4):
    #TRAINING DATA
    train_data[i] = load_dataset("csv", data_files=datafiles[i], delimiter = '\t')
    train_data[i] = train_data[i]['train'].\
        map(lambda x: dict(zip(['id', 'sentence', 'emotion', 'intensity'], 
                                x.values())))
    train_data[i] = train_data[i].map(lambda x: preprocess(x), batched = True)
    train_data[i] = train_data[i].remove_columns(['emotion', 'id'])
    
	#TESTING DATA
    test_data[i] = load_dataset("csv", data_files=datafiles[i+4], delimiter = '\t')
    test_data[i] = test_data[i]['train'].\
        map(lambda x: dict(zip(['id', 'sentence', 'emotion', 'intensity'], 
                                x.values())))
    test_data[i] = test_data[i].map(lambda x: preprocess(x), batched = True)
    test_data[i] = test_data[i].remove_columns(['emotion', 'id'])
    

#TRAINING ARGUMENTS
models = [AutoModelForSequenceClassification.from_pretrained(
    	model_name, 
    	num_labels=1, 
    	ignore_mismatched_sizes=True
    )] * 4


# Training arguments
training_args = TrainingArguments(
    output_dir="test_trainer",
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# initialize Trainers

trainers = [None] * 4

for i in range(4):
    trainers[i] = Trainer(
        model = models[i],
        args = training_args,
        train_dataset = train_data[i],
        eval_dataset = test_data[i]
	)


# train the models
trainers[0].train()
print(trainers[0].evaluate())
#break

trainers[1].train()
print(trainers[1].evaluate())

trainers[2].train()
print(trainers[2].evaluate())

trainers[3].train()
print(trainers[3].evaluate())

print("\n \n TRAINING COMPLETE \n \n")


"""
#test reasonability
angerclassifier = pipeline("sentiment-analysis", model = models[0], tokenizer = tokenizer)

print("i am angry" + str(angerclassifier("i am angry")))
print("i am furious" + str(angerclassifier("i am furious")))
"""

# save the model
models[0].save_pretrained("angerextremitymodel")
models[1].save_pretrained("joyextremitymodel")
models[2].save_pretrained("sadnessextremitymodel")
models[3].save_pretrained("fearextremitymodel")
print("\n \n SAVING COMPLETE \n \n")

# save the tokenizer 
tokenizer.save_pretrained("/Users/juliamargie/Documents/GitHub/persuation2/extremitytokenizer2")
print("\n \n TOKENIZER COMPLETE \n \n")
