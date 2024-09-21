### extremityrun.py
### julia margie
### run previously trained and finetuned roBERTa on Anthropic's persuasion


from transformers import pipeline
from datasets import load_dataset

anger_model = "./angerextremitymodel"
joy_model = "./joyextremitymodel"
sadness_model = "./sadnessextremitymodel"
fear_model = "./fearextremitymodel"

tokenizer2 = "./extremitytokenizer"

angerclassifier = pipeline("sentiment-analysis", model = anger_model, tokenizer = tokenizer2)
joyclassifier = pipeline("sentiment-analysis", model = joy_model, tokenizer = tokenizer2)
sadnessclassifier = pipeline("sentiment-analysis", model = sadness_model, tokenizer = tokenizer2)
fearclassifier = pipeline("sentiment-analysis", model = fear_model, tokenizer = tokenizer2)


# Function to add columns to the dataset with classification and score
def classify_anger(examples):
    results = angerclassifier(examples['argument'])
    examples['anger_score'] = [result['score'] for result in results]
    return examples
def classify_joy(examples):
    results = joyclassifier(examples['argument'])
    examples["joy_score"] = [result["score"] for result in results]
    return examples
def classify_sadness(examples):
    results = sadnessclassifier(examples['argument'])
    examples["sadness_score"] = [result["score"] for result in results]
    return examples
def classify_fear(examples):
    results = fearclassifier(examples['argument'])
    examples["fear_score"] = [result["score"] for result in results]
    return examples

dsfull = load_dataset("Anthropic/persuasion", split='train')
#partial = Dataset.from_dict(dsfull[0:10])
sentiment_dataset = dsfull.map(classify_anger, batched=True)
sentiment_dataset = sentiment_dataset.map(classify_joy, batched=True)
sentiment_dataset = sentiment_dataset.map(classify_sadness, batched=True)
sentiment_dataset = sentiment_dataset.map(classify_fear, batched=True)

toprint = sentiment_dataset.to_pandas()

# Save the DataFrame to a CSV file
toprint.to_csv("with_scores.tsv", sep='\t', index=False)
