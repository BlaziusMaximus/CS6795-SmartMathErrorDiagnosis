import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
  AutoTokenizer,
  get_linear_schedule_with_warmup,
)
import json
import os
import argparse
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

# Your existing Pydantic/Dataset classes would be here or imported
# For this script, we assume they are imported from src.student
from src.student import StudentModel, ErrorDiagnosisDataset
from src.graph import KnowledgeGraph


def train(
  data_file: str,
  model_name: str,
  epochs: int,
  batch_size: int,
  learning_rate: float,
  output_model_path: str,
  test_size: float = 0.2,
):
  """
  The main script to orchestrate the student model training and validation pipeline.
  """
  # --- 1. Load and Split Dataset ---
  print("--- Step 1: Loading and Splitting Dataset ---")
  knowledge_graph = KnowledgeGraph.load_from_json("knowledge_graph.json")
  with open(data_file, "r") as f:
    raw_dataset = json.load(f)

  if not raw_dataset:
    print("Dataset is empty. Exiting.")
    return

  print(f"\nSuccessfully loaded {len(raw_dataset)} total examples.")

  # Create the label map from the full dataset BEFORE splitting
  all_failure_ids = sorted(
    list(set(item["failure_concept_id"] for item in raw_dataset))
  )
  label_map = {label: i for i, label in enumerate(all_failure_ids)}
  num_labels = len(label_map)

  # Split the dataset
  train_texts, val_texts = train_test_split(
    raw_dataset,
    test_size=test_size,
    random_state=42,
    stratify=[label_map[item["failure_concept_id"]] for item in raw_dataset],
  )
  print(
    f"Split dataset into {len(train_texts)} training examples and {len(val_texts)} validation examples."
  )

  # --- 2. Dataset and DataLoader Setup ---
  print("\n--- Step 2: Preparing Datasets and DataLoaders ---")
  os.makedirs(output_model_path, exist_ok=True)
  with open(os.path.join(output_model_path, "label_map.json"), "w") as f:
    json.dump(label_map, f)

  tokenizer = AutoTokenizer.from_pretrained(model_name);

  train_dataset = ErrorDiagnosisDataset(
    train_texts, tokenizer, label_map, knowledge_graph
  )
  val_dataset = ErrorDiagnosisDataset(
    val_texts, tokenizer, label_map, knowledge_graph
  )

  train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
  )
  val_dataloader = DataLoader(
    val_dataset, batch_size=batch_size
  )  # No need to shuffle validation data

  # --- 3. Model, Optimizer, and Scheduler Setup ---
  print("\n--- Step 3: Initializing Model and Optimizer ---")
  model = StudentModel(num_labels=num_labels, model_name=model_name)
  optimizer = AdamW(model.parameters(), lr=learning_rate)
  total_steps = len(train_dataloader) * epochs
  scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
  )
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)
  print(f"Training on device: {device}")

  # --- 4. Training and Validation Loop ---
  print("\n--- Step 4: Starting Training & Validation Loop ---")
  progress_bar = tqdm(range(total_steps))
  best_val_loss = float("inf")

  for epoch in range(epochs):
    # -- Training Phase --
    model.train()
    total_train_loss = 0
    for batch in train_dataloader:
      model.zero_grad()
      input_ids = batch["input_ids"].to(device)
      attention_mask = batch["attention_mask"].to(device)
      labels = batch["labels"].to(device)
      outputs = model(
        input_ids=input_ids, attention_mask=attention_mask, labels=labels
      )
      loss = outputs.loss
      total_train_loss += loss.item()
      loss.backward()
      optimizer.step()
      scheduler.step()
      progress_bar.update(1)

    avg_train_loss = total_train_loss / len(train_dataloader)

    # -- Validation Phase --
    model.eval()  # Set the model to evaluation mode
    total_val_loss = 0
    for batch in val_dataloader:
      with torch.no_grad():  # No need to calculate gradients for validation
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(
          input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        loss = outputs.loss
        total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_dataloader)

    print(
      f"\nEpoch {epoch + 1}/{epochs} | Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f}"
    )

    # -- Save the best model --
    if avg_val_loss < best_val_loss:
      print(
        f"Validation loss improved ({best_val_loss:.4f} --> {avg_val_loss:.4f}). Saving model..."
      )
      best_val_loss = avg_val_loss
      model.bert.save_pretrained(output_model_path)
      tokenizer.save_pretrained(output_model_path)

  print("\nTraining complete.")
  print(f"Best validation loss: {best_val_loss:.4f}")
  print(f"Best model saved to {output_model_path}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Train and validate the student model."
  )
  # Add arguments as before...
  parser.add_argument(
    "--data-file", type=str, default="./data/synthetic_dataset.json"
  )
  parser.add_argument(
    "--model-name", type=str, default="distilbert-base-uncased"
  )
  parser.add_argument("--epochs", type=int, default=4)
  parser.add_argument("--batch-size", type=int, default=8)
  parser.add_argument("--learning-rate", type=float, default=5e-5)
  parser.add_argument(
    "--output-model-path", type=str, default="./student_model_finetuned"
  )
  parser.add_argument(
    "--test-size",
    type=float,
    default=0.2,
    help="Proportion of the dataset to use for validation.",
  )

  args = parser.parse_args()

  train(
    data_file=args.data_file,
    model_name=args.model_name,
    epochs=args.epochs,
    batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    output_model_path=args.output_model_path,
    test_size=args.test_size,
  )
