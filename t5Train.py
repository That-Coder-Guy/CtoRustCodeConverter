from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,Seq2SeqTrainer, Seq2SeqTrainingArguments

model_name = "Salesforce/codet5-small"

print("Loading tokenizer...")
## this tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "Salesforce/codet5-small",
    use_fast=False
)

print("Loading model...")
## this model used in the training
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

print("Model loaded successfully")


dataset = load_dataset("json", data_files={
    "train": "/homes/edermar/KSU_Courses/CIS_730_Principles_Of_AI/GP/training_model/TrainingValTestSets/repaired_c_rust_train.jsonl"
})

print(dataset["train"][0])

## at this point we have a raw dataset. 


# Makes in the dataset and converst it to 
# input output. 
# T5 expects input: output. Instead of C and Rust
def format_example(example):
    return {
        "input": "translate C to Rust:\n" + example["C"],
        "output": example["Rust"]
    }

dataset = dataset.map(format_example)

#print both the training and output. 
print(dataset["train"][0]["input"])
print(dataset["train"][0]["output"])

"""
Begin training
"""
#------------------ preprosess ----------------- 
def preprocess(example):
    inputs = tokenizer(
        example["input"],
        truncation=True,
        max_length=512
    )

    labels = tokenizer(
        example["output"],
        truncation=True,
        max_length=512
    )

    inputs["labels"] = labels["input_ids"]
    return inputs

dataset = dataset.map(preprocess, remove_columns=dataset["train"].column_names)

#----------------select subset and train-------------
#select range
small_train = dataset["train"].select(range(492))

training_args = Seq2SeqTrainingArguments(
    output_dir = "./output_train",
    per_device_train_batch_size = 1,
    num_train_epochs=3,
    logging_steps = 10
)
#ambigous name?
trainer = Seq2SeqTrainer(
    model = model,
    args = training_args,
    train_dataset = small_train,
    tokenizer = tokenizer

)

trainer.train()
# Save model 
print("---Finished training--- Saving")
trainer.save_model("./TrainedModels/c2rust_model_small_smallset_3_e")
tokenizer.save_pretrained("./TrainedModels/c2rust_model_small_smallset_3_e")
