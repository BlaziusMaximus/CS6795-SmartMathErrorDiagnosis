import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from src.student import StudentModel, ErrorDiagnosisDataset
import json
import os
import argparse
from tqdm.auto import tqdm  # For a nice progress bar


def train(
  data_file: str,
  model_name: str,
  epochs: int,
  batch_size: int,
  learning_rate: float,
  output_model_path: str,
):
  """
  The main script to orchestrate the model training pipeline.
  """
  # --- 1. Load Dataset ---
  print("--- Step 1: Loading Dataset ---")
  with open(data_file, "r") as f:
    raw_dataset = json.load(f)

  if not raw_dataset:
    print("Dataset is empty. Exiting.")
    return

  print(f"\nSuccessfully loaded {len(raw_dataset)} training examples.")

  # --- 2. Dataset and DataLoader Setup ---
  print("\n--- Step 2: Preparing Dataset and DataLoader ---")

  # Create a mapping from concept ID strings to integer labels
  all_failure_ids = sorted(
    list(set(item["failure_concept_id"] for item in raw_dataset))
  )
  label_map = {label: i for i, label in enumerate(all_failure_ids)}
  num_labels = len(label_map)

  print(f"Found {num_labels} unique failure concepts to classify.")
  # Save the label map for later use during inference
  os.makedirs(output_model_path, exist_ok=True)
  with open(os.path.join(output_model_path, "label_map.json"), "w") as f:
    json.dump(label_map, f)

  # Initialize tokenizer and create the PyTorch Dataset
  tokenizer = BertTokenizer.from_pretrained(model_name)
  train_dataset = ErrorDiagnosisDataset(raw_dataset, tokenizer, label_map)

  # Create the DataLoader
  train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
  )

  # --- 3. Model, Optimizer, and Scheduler Setup ---
  print("\n--- Step 3: Initializing Model and Optimizer ---")

  # Initialize the Student Model
  model = StudentModel(num_labels=num_labels, model_name=model_name)

  # Set up the optimizer
  optimizer = AdamW(model.parameters(), lr=learning_rate)

  # Set up the learning rate scheduler
  total_steps = len(train_dataloader) * epochs
  scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
  )

  # Check for GPU availability
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)
  print(f"Training on device: {device}")

  # --- 4. Training Loop ---
  print("\n--- Step 4: Starting Training Loop ---")

  # Use tqdm for a user-friendly progress bar
  progress_bar = tqdm(range(total_steps))

  model.train()  # Set the model to training mode
  for epoch in range(epochs):
    total_train_loss = 0
    for batch in train_dataloader:
      # Move batch to the same device as the model
      input_ids = batch["input_ids"].to(device)
      attention_mask = batch["attention_mask"].to(device)
      labels = batch["labels"].to(device)

      # Clear any previously calculated gradients
      model.zero_grad()

      # Perform a forward pass
      outputs = model(
        input_ids=input_ids, attention_mask=attention_mask, labels=labels
      )

      # The loss is the first item in the outputs tuple
      loss = outputs.loss

      total_train_loss += loss.item()

      # Perform a backward pass to calculate gradients
      loss.backward()

      # Update weights
      optimizer.step()

      # Update the learning rate
      scheduler.step()

      progress_bar.update(1)

    avg_train_loss = total_train_loss / len(train_dataloader)
    print(
      f"\nEpoch {epoch + 1}/{epochs} | Average Training Loss: {avg_train_loss:.4f}"
    )

  print("\nTraining complete.")

  # --- 5. Save the Fine-Tuned Model ---
  print(f"Saving model to {output_model_path}")
  model.bert.save_pretrained(output_model_path)
  tokenizer.save_pretrained(output_model_path)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Train the student model on synthetic error data."
  )
  parser.add_argument(
    "--data-file",
    type=str,
    default="./data/synthetic_dataset.json",
    help="The path to the generated dataset.",
  )
  parser.add_argument(
    "--model-name",
    type=str,
    default="bert-base-uncased",
    help="The name of the pre-trained model to use.",
  )
  parser.add_argument(
    "--epochs", type=int, default=3, help="The number of training epochs."
  )
  parser.add_argument(
    "--batch-size", type=int, default=8, help="The training batch size."
  )
  parser.add_argument(
    "--learning-rate", type=float, default=2e-5, help="The learning rate."
  )
  parser.add_argument(
    "--output-model-path",
    type=str,
    default="./student_model_finetuned",
    help="The path to save the fine-tuned model.",
  )
  args = parser.parse_args()

  train(
    data_file=args.data_file,
    model_name=args.model_name,
    epochs=args.epochs,
    batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    output_model_path=args.output_model_path,
  )
