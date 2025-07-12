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
      "Starting dataset generation from node:"
      f"'{start_node_id}' with max depth: {self.max_depth}"
    )

    start_node = self.graph.get_node(start_node_id)
    assert start_node, f"Start node '{start_node_id}' not found in graph."

    problem_example = start_node.problem_and_solution.problem
    correct_solution = start_node.problem_and_solution.solution

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

    teacher_response = self.teacher.generate_synthetic_errors(
      problem_example=problem_example,
      correct_solution=correct_solution,
      failure_concept=prereq_node,
    )

    if teacher_response and teacher_response.is_valid_error:
      for result in teacher_response.generated_solutions:
        print(f"\tResult: Step {result.step_number}")
        print(
          f"\tSolution:\n\t\t{'\n\t\t'.join([step for step in result.incorrect_solution])}"
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
      print(
        "    - Implausible error or invalid response. "
        f"Pruning this path. Reasoning: {teacher_response.reasoning if teacher_response else 'No response'}"
      )

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
          f"Stopping at node '{current_node_id}' "
          f"(depth {current_depth}) due to max depth."
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
