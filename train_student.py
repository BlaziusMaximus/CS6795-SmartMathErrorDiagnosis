import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from src.graph import KnowledgeGraph
from src.teacher import TeacherModel
from src.data_generator import DataGenerator
from src.student import StudentModel, ErrorDiagnosisDataset
import json
import os
from tqdm.auto import tqdm  # For a nice progress bar


def train():
  """
  The main script to orchestrate the data generation and model training pipeline.
  """
  # --- 1. Configuration ---
  # In a real project, these would come from a config file or argparse
  START_NODE_ID = (
    "1023"  # "Solving Systems of Equations Using Inverse Matrices"
  )
  MAX_TRAVERSAL_DEPTH = 2
  MODEL_NAME = "bert-base-uncased"
  EPOCHS = 3
  BATCH_SIZE = 8
  LEARNING_RATE = 2e-5
  OUTPUT_MODEL_PATH = "./student_model_finetuned"

  # --- 2. Data Generation ---
  print("--- Step 1: Generating Synthetic Dataset ---")
  knowledge_graph = KnowledgeGraph.load_from_json("knowledge_graph.json")
  teacher = TeacherModel()
  data_generator = DataGenerator(knowledge_graph, teacher)

  # Generate the raw dataset
  raw_dataset = data_generator.generate_error_dataset(
    start_node_id=START_NODE_ID, max_depth=MAX_TRAVERSAL_DEPTH
  )

  if not raw_dataset:
    print("Dataset generation failed or produced no data. Exiting.")
    return

  print(f"\nSuccessfully generated {len(raw_dataset)} training examples.")

  # --- 3. Dataset and DataLoader Setup ---
  print("\n--- Step 2: Preparing Dataset and DataLoader ---")

  # Create a mapping from concept ID strings to integer labels
  all_failure_ids = sorted(
    list(set(item["failure_concept_id"] for item in raw_dataset))
  )
  label_map = {label: i for i, label in enumerate(all_failure_ids)}
  num_labels = len(label_map)

  print(f"Found {num_labels} unique failure concepts to classify.")
  # Save the label map for later use during inference
  os.makedirs(OUTPUT_MODEL_PATH, exist_ok=True)
  with open(os.path.join(OUTPUT_MODEL_PATH, "label_map.json"), "w") as f:
    json.dump(label_map, f)

  # Initialize tokenizer and create the PyTorch Dataset
  tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
  train_dataset = ErrorDiagnosisDataset(raw_dataset, tokenizer, label_map)

  # Create the DataLoader
  train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
  )

  # --- 4. Model, Optimizer, and Scheduler Setup ---
  print("\n--- Step 3: Initializing Model and Optimizer ---")

  # Initialize the Student Model
  model = StudentModel(num_labels=num_labels, model_name=MODEL_NAME)

  # Set up the optimizer
  optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

  # Set up the learning rate scheduler
  total_steps = len(train_dataloader) * EPOCHS
  scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
  )

  # Check for GPU availability
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)
  print(f"Training on device: {device}")

  # --- 5. Training Loop (To be implemented in the next step) ---
  print("\n--- Step 4: Starting Training Loop ---")

  # Use tqdm for a user-friendly progress bar
  progress_bar = tqdm(range(total_steps))

  model.train()  # Set the model to training mode
  for epoch in range(EPOCHS):
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
      f"\nEpoch {epoch + 1}/{EPOCHS} | Average Training Loss: {avg_train_loss:.4f}"
    )

  print("\nTraining complete.")

  # --- 6. Save the Fine-Tuned Model (To be implemented) ---
  print(f"Saving model to {OUTPUT_MODEL_PATH}")
  model.bert.save_pretrained(OUTPUT_MODEL_PATH)
  tokenizer.save_pretrained(OUTPUT_MODEL_PATH)


if __name__ == "__main__":
  train()
