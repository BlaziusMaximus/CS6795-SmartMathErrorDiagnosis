import argparse
import json
from pathlib import Path
import concurrent.futures
from concurrent.futures import as_completed
from src.graph import KnowledgeGraph
from src.teacher import TeacherModel
from src.failure_data_generator import FailureDataGenerator
from src.rate_limiter import ThreadSafeRateLimiter


def generate_failures(
  start_node_id: str | None,
  output_file: str,
  max_workers: int,
):
  """
  Generates a synthetic dataset by processing start nodes in parallel.
  For each start node, it analyzes all problems against all descendants.
  """
  print("--- Generating Synthetic Failure Dataset (Parallel) ---")
  knowledge_graph = KnowledgeGraph.load_from_json("knowledge_graph.json")
  rate_limiter = ThreadSafeRateLimiter(max_requests=999, period_seconds=60)
  teacher = TeacherModel(rate_limiter=rate_limiter)
  generator = FailureDataGenerator(knowledge_graph, teacher)
  full_dataset = []

  if start_node_id:
    nodes_to_process = [knowledge_graph.get_node(start_node_id)]
  else:
    print("No start node provided. Finding all nodes with problems...")
    nodes_to_process = [
      node
      for node in knowledge_graph.nodes.values()
      if node.problems_and_solutions
    ]

  # Filter out any nodes that weren't found
  nodes_to_process = [node for node in nodes_to_process if node]

  if not nodes_to_process:
    print("No valid start nodes found to process.")
    return

  print(f"Found {len(nodes_to_process)} start nodes to process.")

  with concurrent.futures.ThreadPoolExecutor(
    max_workers=max_workers
  ) as executor:
    # Create a future for each start node
    future_to_node = {
      executor.submit(generator.generate_errors_for_node, node): node
      for node in nodes_to_process
    }

    for future in as_completed(future_to_node):
      node = future_to_node[future]
      try:
        data = future.result()
        if data:
          print(
            f"--- Finished processing for node: {node.name}, found {len(data)} examples ---"
          )
          full_dataset.extend(data)
        else:
          print(
            f"--- Finished processing for node: {node.name}, no examples generated ---"
          )
      except Exception as exc:
        print(f"Node {node.name} generated an exception: {exc}")

  if not full_dataset:
    print("Dataset generation failed or produced no data. Exiting.")
    return

  output_path = Path(output_file)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with open(output_path, "w") as f:
    json.dump(full_dataset, f, indent=2)

  print(
    f"\nSuccessfully generated {len(full_dataset)} total training examples."
  )
  print(f"Dataset saved to {output_path}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Generate synthetic error data for the student model in parallel."
  )
  parser.add_argument("--start-node", type=str, default=None)
  # The max_depth argument is no longer needed with this design
  parser.add_argument(
    "--output-file", type=str, default="./data/synthetic_dataset.json"
  )
  parser.add_argument("--max-workers", type=int, default=10)
  args = parser.parse_args()

  generate_failures(
    start_node_id=args.start_node,
    output_file=args.output_file,
    max_workers=args.max_workers,
  )
