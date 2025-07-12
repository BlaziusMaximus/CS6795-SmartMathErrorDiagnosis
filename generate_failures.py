import argparse
import json
from pathlib import Path
from src.graph import KnowledgeGraph
from src.teacher import TeacherModel
from src.failure_data_generator import FailureDataGenerator


def generate_failures(
  start_node_id: str | None, max_depth: int, output_file: str
):
  """
  Generates a synthetic dataset of failures for training the student model.

  If a start_node_id is provided, it generates data for that specific
  sub-tree. Otherwise, it finds all nodes with a problem definition and
  generates data for each of them.

  Args:
      start_node_id: The ID of the concept node to start the traversal from.
                     If None, generate for all problem nodes.
      max_depth: The maximum depth to traverse the knowledge graph.
      output_file: The path to save the generated dataset.
  """
  print("--- Generating Synthetic Failure Dataset ---")
  knowledge_graph = KnowledgeGraph.load_from_json("knowledge_graph.json")
  teacher = TeacherModel()
  failure_data_generator = FailureDataGenerator(knowledge_graph, teacher)
  full_dataset = []

  if start_node_id:
    start_nodes = [start_node_id]
  else:
    print("No start node provided. Finding all problem nodes in the graph...")
    # Find all nodes that have a problem defined
    start_nodes = [
      node.id
      for node in knowledge_graph.nodes.values()
      if node.problems_and_solutions
    ]
    print(f"Found {len(start_nodes)} problem nodes to process.")

  for i, node_id in enumerate(start_nodes):
    print(
      f"\n--- Processing Start Node {i + 1}/{len(start_nodes)}: {node_id} ---"
    )
    # Generate the raw dataset for the current start node
    raw_dataset = failure_data_generator.generate_error_dataset(
      start_node_id=node_id, max_depth=max_depth
    )
    if raw_dataset:
      full_dataset.extend(raw_dataset)

  if not full_dataset:
    print("Dataset generation failed or produced no data. Exiting.")
    return

  # Ensure the output directory exists
  output_path = Path(output_file)
  output_path.parent.mkdir(parents=True, exist_ok=True)

  # Save the dataset to a file
  with open(output_path, "w") as f:
    json.dump(full_dataset, f, indent=2)

  print(f"\nSuccessfully generated {len(full_dataset)} training examples.")
  print(f"Dataset saved to {output_path}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Generate synthetic error data for the student model."
  )
  parser.add_argument(
    "--start-node",
    type=str,
    default=None,
    help="The ID of the concept node to start from. If not provided, runs on all problem nodes.",
  )
  parser.add_argument(
    "--max-depth",
    type=int,
    default=2,
    help="The maximum depth to traverse the knowledge graph.",
  )
  parser.add_argument(
    "--output-file",
    type=str,
    default="./data/synthetic_dataset.json",
    help="The path to save the generated dataset.",
  )
  args = parser.parse_args()

  generate_failures(
    start_node_id=args.start_node,
    max_depth=args.max_depth,
    output_file=args.output_file,
  )
