from collections import deque

from typing import Deque, Dict, List, Set, Tuple
from .graph import KnowledgeGraph, PrerequisiteEdge
from .teacher import TeacherModel


class DataGenerator:
  """
  Orchestrates the generation of a synthetic error dataset by intelligently
  traversing the knowledge graph.
  """

  def __init__(
    self, knowledge_graph: KnowledgeGraph, teacher_model: TeacherModel
  ):
    """
    Initializes the DataGenerator.

    Args:
        knowledge_graph: An instance of the loaded knowledge graph.
        teacher_model: An instance of the TeacherModel for generating errors.
    """
    self.graph = knowledge_graph
    self.teacher = teacher_model
    # This dictionary holds the manually created problems and correct solutions.
    # The key is the concept ID.
    self.problems_and_solutions = {
      "multiply_2d_by_1d": {
        "problem": "Calculate 17 * 5",
        "solution": (
          "Step 1: Multiply the ones digit: 7 * 5 = 35. Write down 5, carry over 3.\n"
          "Step 2: Multiply the tens digit: 1 * 5 = 5.\n"
          "Step 3: Add the carry-over: 5 + 3 = 8.\n"
          "Step 4: Combine the results. Final Answer: 85."
        ),
      }
    }

  def _initialize_traversal(
    self, start_node_id: str
  ) -> Tuple[List[Dict], Deque[Tuple[str, int]], Set[str], str, str]:
    """
    Initializes the data structures and retrieves the problem context for traversal.
    """
    training_dataset: List[Dict] = []
    queue: Deque[Tuple[str, int]] = deque(
      [(start_node_id, 0)]
    )  # Each item: (node_id, depth)
    visited: Set[str] = {start_node_id}

    print(
      f"Starting dataset generation from node: '{start_node_id}' with max depth: {self.max_depth}"
    )

    main_problem_context = self.problems_and_solutions.get(start_node_id)
    if not main_problem_context:
      print(
        f"Error: No problem/solution found for start node '{start_node_id}'."
      )
      # Return empty collections with correct types for consistency
      return [], deque(), set(), "", ""

    problem_example = main_problem_context["problem"]
    correct_solution = main_problem_context["solution"]
    return training_dataset, queue, visited, problem_example, correct_solution

  def _process_prerequisite(
    self,
    prereq_edge: PrerequisiteEdge,
    current_depth: int,
    problem_example: str,
    correct_solution: str,
    visited: Set[str],
    queue: Deque[Tuple[str, int]],
    training_dataset: List[Dict],
    start_node_id: str,
  ):
    """
    Processes a single prerequisite node, calls the teacher model, and updates data.
    """
    prereq_node_id = prereq_edge.concept_id
    if prereq_node_id in visited:
      print(
        f"  -> Already visited prerequisite node '{prereq_node_id}'. Skipping."
      )
      return

    prereq_node = self.graph.get_node(prereq_node_id)
    if not prereq_node:
      print(
        f"Warning: Prerequisite node '{prereq_node_id}' not found. Skipping."
      )
      return

    print(f"  -> Checking prerequisite: '{prereq_node.name}'")

    analysis_results = self.teacher.generate_synthetic_errors(
      problem_example=problem_example,
      correct_solution=correct_solution,
      failure_concept=prereq_node,
    )

    for result in analysis_results:
      print(f"\tResult: Step {result.step_number}")
      print(
        f"\tSolution:\n{'\t\t'.join([step + '\n' for step in result.incorrect_solution])}"
      )
      training_dataset.append(
        {
          "target_concept_id": start_node_id,
          "failure_concept_id": prereq_node_id,
          "incorrect_solution": result.incorrect_solution,
          "incorrect_step_number": result.step_number,
        }
      )

      if prereq_node_id not in visited:
        visited.add(prereq_node_id)
        queue.append((prereq_node_id, current_depth + 1))
    else:
      print("    - Implausible error. Pruning this path.")

  def generate_error_dataset(
    self, start_node_id: str, max_depth: int = 3
  ) -> list[dict]:
    """
    Generates a dataset of synthetic errors by traversing the knowledge graph.
    """
    self.max_depth = max_depth  # Store max_depth as an instance variable
    (
      training_dataset,
      queue,
      visited,
      problem_example,
      correct_solution,
    ) = self._initialize_traversal(start_node_id)

    if training_dataset is None:  # Handle case where initialization failed
      return []

    while queue:
      current_node_id, current_depth = queue.popleft()
      current_node = self.graph.get_node(current_node_id)

      if not current_node:
        print(
          f"Warning: Node '{current_node_id}' not found in graph. Skipping."
        )
        continue

      if current_depth >= self.max_depth:
        print(
          f"Stopping at node '{current_node_id}' (depth {current_depth}) due to max depth."
        )
        continue

      print(
        f"\nProcessing node '{current_node.name}' at depth {current_depth}..."
      )

      for prereq_edge in current_node.prerequisites:
        self._process_prerequisite(
          prereq_edge,
          current_depth,
          problem_example,
          correct_solution,
          visited,
          queue,
          training_dataset,
          start_node_id,
        )

    print("\nDataset generation complete.")
    return training_dataset
