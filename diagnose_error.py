import argparse
import json
import torch
from transformers import AutoTokenizer
from src.student import StudentModel


def diagnose(model_dir: str, problem_example: str, incorrect_solution: str):
  """
  Diagnoses the conceptual error in an incorrect solution using the fine-tuned
  StudentModel.

  Args:
      model_dir: The directory where the fine-tuned model is saved.
      problem_example: The original problem text.
      incorrect_solution: The incorrect solution text to diagnose.
  """
  # --- 1. Load Model, Tokenizer, and Label Map ---
  print(f"--- Loading model from {model_dir} ---")
  with open(f"{model_dir}/label_map.json", "r") as f:
    label_map = json.load(f)

  num_labels = len(label_map)
  model = StudentModel(num_labels=num_labels, model_name=model_dir)
  tokenizer = AutoTokenizer.from_pretrained(model_dir)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)
  model.eval()

  # Create the inverse mapping from integer to concept_id
  inverse_label_map = {v: k for k, v in label_map.items()}

  # --- 2. Tokenize the Input ---
  print("\n--- Tokenizing Input ---")
  text = f"Problem: {problem_example} Solution: {incorrect_solution}"
  encoding = tokenizer.encode_plus(
    text,
    add_special_tokens=True,
    max_length=256,
    padding="max_length",
    truncation=True,
    return_attention_mask=True,
    return_tensors="pt",
  )
  input_ids = encoding["input_ids"].to(device)
  attention_mask = encoding["attention_mask"].to(device)

  # --- 3. Get Model Prediction ---
  print("--- Getting Model Prediction ---")
  with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

  # --- 4. Process Prediction ---
  probabilities = torch.softmax(logits, dim=1)
  confidence_tensor, predicted_label_tensor = torch.max(probabilities, dim=1)
  predicted_label_int = predicted_label_tensor.item()
  confidence = confidence_tensor.item()
  predicted_concept_id = inverse_label_map[predicted_label_int]

  # --- 5. Print Diagnosis ---
  print("\n--- Diagnosis Result ---")
  print(f"Predicted Failure Concept ID: {predicted_concept_id}")
  print(f"Confidence: {confidence:.4f}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Diagnose the conceptual error in a given incorrect solution."
  )
  parser.add_argument(
    "--model-dir",
    type=str,
    default="./student_model_finetuned",
    help="Directory containing the fine-tuned model, tokenizer, and label map.",
  )
  parser.add_argument(
    "--problem",
    type=str,
    required=True,
    help="The original problem text.",
  )
  parser.add_argument(
    "--solution",
    type=str,
    required=True,
    help="The incorrect solution text to be diagnosed.",
  )
  args = parser.parse_args()

  diagnose(
    model_dir=args.model_dir,
    problem_example=args.problem,
    incorrect_solution=args.solution,
  )
