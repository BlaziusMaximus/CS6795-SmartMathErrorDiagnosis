import argparse
import json
import torch
from transformers import AutoTokenizer
from src.student import StudentModel
from src.graph import KnowledgeGraph
from tqdm import tqdm


def evaluate(model_dir: str, dataset_path: str):
  """
  Evaluates the fine-tuned StudentModel on a given dataset.

  Args:
      model_dir: The directory where the fine-tuned model is saved.
      dataset_path: The path to the JSON dataset file to evaluate.
  """
  # --- 1. Load Model, Tokenizer, and Label Map ---
  print(f"--- Loading model from {model_dir} ---")
  knowledge_graph = KnowledgeGraph.load_from_json("knowledge_graph.json")
  with open(f"{model_dir}/label_map.json", "r") as f:
    label_map = json.load(f)
  with open(dataset_path, "r") as f:
    dataset = json.load(f)

  num_labels = len(label_map)
  model = StudentModel(num_labels=num_labels, model_name=model_dir)
  tokenizer = AutoTokenizer.from_pretrained(model_dir)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)
  model.eval()

  # Create the inverse mapping from integer to concept_id
  inverse_label_map = {v: k for k, v in label_map.items()}
  correct_predictions = 0

  print(f"\n--- Evaluating on {len(dataset)} examples from {dataset_path} ---")
  for item in tqdm(dataset, desc="Evaluating"):
    # --- 2. Tokenize the Input ---
    problem_example = item["problem_example"]
    incorrect_solution = item["incorrect_solution"]
    target_concept_id = item["target_concept_id"]
    true_failure_id = item["failure_concept_id"]

    descendants = knowledge_graph.get_all_descendants(target_concept_id)
    descendant_names = ", ".join([d.name for d, _ in descendants])
    prereq_context = f"Relevant Concepts: {descendant_names}"
    text = f"Problem: {problem_example} [SEP] {prereq_context} [SEP] Solution: {incorrect_solution}"

    encoding = tokenizer.encode_plus(
      text,
      add_special_tokens=True,
      max_length=512,
      padding="max_length",
      truncation=True,
      return_attention_mask=True,
      return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # --- 3. Get Model Prediction ---
    with torch.no_grad():
      outputs = model(input_ids=input_ids, attention_mask=attention_mask)
      logits = outputs.logits

    # --- 4. Process Prediction ---
    probabilities = torch.softmax(logits, dim=1)
    _, predicted_label_tensor = torch.max(probabilities, dim=1)
    predicted_label_int = predicted_label_tensor.item()
    predicted_concept_id = inverse_label_map[predicted_label_int]

    if predicted_concept_id == true_failure_id:
      correct_predictions += 1

  # --- 5. Print Diagnosis ---
  accuracy = (correct_predictions / len(dataset)) * 100
  print("\n--- Evaluation Result ---")
  print(f"Total Examples: {len(dataset)}")
  print(f"Correct Predictions: {correct_predictions}")
  print(f"Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Evaluate the student model on a dataset."
  )
  parser.add_argument(
    "--model-dir",
    type=str,
    default="./student_model_finetuned",
    help="Directory containing the fine-tuned model, tokenizer, and label map.",
  )
  parser.add_argument(
    "--dataset-path",
    type=str,
    default="./failure_data/synthetic_dataset.json",
    help="Path to the dataset file to evaluate.",
  )
  args = parser.parse_args()

  evaluate(model_dir=args.model_dir, dataset_path=args.dataset_path)
