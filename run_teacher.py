from src.graph import KnowledgeGraph
from src.teacher import TeacherModel
from src.data_generator import DataGenerator


def main():
  """
  A simple script to test the full data generation pipeline.
  """
  print("Initializing components...")
  try:
    # Load the knowledge graph from our JSON file
    knowledge_graph = KnowledgeGraph.load_from_json("knowledge_graph.json")

    # Initialize the teacher model
    teacher = TeacherModel()

    # Initialize the data generator
    data_generator = DataGenerator(knowledge_graph, teacher)

  except (ValueError, FileNotFoundError) as e:
    print(f"Error during initialization: {e}")
    return

  # Start the data generation process from our main problem concept
  start_concept_id = "multiply_2d_by_1d"

  # Run the dataset generation
  dataset = data_generator.generate_error_dataset(start_concept_id)

  # For now, just print the result
  print("\n--- Generated Dataset ---")
  if not dataset:
    print("No plausible errors were generated in this run.")
  else:
    for i, entry in enumerate(dataset):
      print(f"\n--- Entry {i + 1} ---")
      print(f"Target Concept: {entry['target_concept_id']}")
      print(f"Failure Concept: {entry['failure_concept_id']}")
      print("Incorrect Solution:")
      print(entry["incorrect_solution"])
      print("-" * 20)


if __name__ == "__main__":
  main()
