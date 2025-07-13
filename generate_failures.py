import argparse
import json
from pathlib import Path
import concurrent.futures
from src.graph import KnowledgeGraph
from src.teacher import TeacherModel
from src.failure_data_generator import FailureDataGenerator
from src.rate_limiter import ThreadSafeRateLimiter


def generate_data_for_node(
  node_id: str, generator: FailureDataGenerator, max_depth: int
) -> list[dict]:
  """Wrapper function to run data generation for a single node."""
  print(f"--- Starting processing for node: {node_id} ---")
  dataset = generator.generate_error_dataset(
    start_node_id=node_id, max_depth=max_depth
  )
  print(
    f"--- Finished processing for node: {node_id}, found {len(dataset)} examples ---"
  )
  return dataset


def generate_failures(
  start_node_id: str | None,
  max_depth: int,
  output_file: str,
  max_workers: int,
):
  """
  Generates a synthetic dataset of failures in parallel using a thread pool.

  Args:
      start_node_id: The ID of the concept node to start from. If None,
                     generate for all problem nodes.
      max_depth: The maximum depth to traverse the knowledge graph.
      output_file: The path to save the generated dataset.
      max_workers: The maximum number of threads to use.
  """
  print("--- Generating Synthetic Failure Dataset (Parallel) ---")
  knowledge_graph = KnowledgeGraph.load_from_json("knowledge_graph.json")
  # Rate limiter: 1000 requests per minute (a bit below the API limit)
  rate_limiter = ThreadSafeRateLimiter(max_requests=999, period_seconds=60)
  teacher = TeacherModel(rate_limiter=rate_limiter)
  generator = FailureDataGenerator(knowledge_graph, teacher)
  full_dataset = []

  if start_node_id:
    start_nodes = [start_node_id]
  else:
    print("No start node provided. Finding all problem nodes in the graph...")
    start_nodes = [
      node.id
      for node in knowledge_graph.nodes.values()
      if node.problems_and_solutions
    ]
    print(f"Found {len(start_nodes)} problem nodes to process.")

  with concurrent.futures.ThreadPoolExecutor(
    max_workers=max_workers
  ) as executor:
    # Map each node ID to a future
    future_to_node = {
      executor.submit(
        generate_data_for_node, node_id, generator, max_depth
      ): node_id
      for node_id in start_nodes
    }
    for future in concurrent.futures.as_completed(future_to_node):
      node_id = future_to_node[future]
      try:
        data = future.result()
        if data:
          full_dataset.extend(data)
      except Exception as exc:
        print(f"Node {node_id} generated an exception: {exc}")

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
  parser.add_argument(
    "--start-node",
    type=str,
    default=None,
    help="The ID of the concept node to start from.",
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
    default="./failure_data/synthetic_dataset.json",
    help="The path to save the generated dataset.",
  )
  parser.add_argument(
    "--max-workers",
    type=int,
    default=20,
    help="The maximum number of worker threads.",
  )
  args = parser.parse_args()

  generate_failures(
    start_node_id=args.start_node,
    max_depth=args.max_depth,
    output_file=args.output_file,
    max_workers=args.max_workers,
  )
