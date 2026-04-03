from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load trained model
model_path = "./TrainedModels/c2rust_model"

print("Loading trained model...")
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load test dataset (your new jsonl)
dataset = load_dataset("json", data_files={
    "test": "TraningValTestSets/repaired_c_rust_test.jsonl"   # ← change name if needed
})

test_data = dataset["test"]

print(f"Loaded {len(test_data)} test samples\n")

# Loop through a few examples
for i in range(5):   # test first 5 examples
    example = test_data[i]

    c_code = example["C"]
    expected_rust = example["Rust"]

    prompt = "translate C to Rust:\n" + c_code

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_length=256,
        num_beams=4,
        early_stopping=True
    )

    predicted_rust = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("="*60)
    print(f"Example {i+1}")
    print("\nC CODE:\n", c_code)
    print("\nEXPECTED RUST:\n", expected_rust)
    print("\nPREDICTED RUST:\n", predicted_rust)
    print("="*60 + "\n")