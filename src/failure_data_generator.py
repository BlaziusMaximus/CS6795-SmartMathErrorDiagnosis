from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List
from .graph import KnowledgeGraph, ConceptNode
from .teacher import TeacherModel
import time

# These constants can now be simplified as we apply them differently
SOLUTIONS_PER_PREREQ = 10
REPEATS_PER_PREREQ = 10


class FailureDataGenerator:
  """
  Orchestrates the generation of a synthetic error dataset by analyzing
  a given node's problem portfolio against its entire prerequisite subtree.
  """

  def __init__(
    self, knowledge_graph: KnowledgeGraph, teacher_model: TeacherModel
  ):
    """
    Initializes the FailureDataGenerator.
    """
    self.graph = knowledge_graph
    self.teacher = teacher_model

  def generate_errors_for_node(self, start_node: ConceptNode) -> list[dict]:
    """
    Generates a dataset for a single start node by analyzing all its problems
    against all of its descendant prerequisites.

    Args:
        start_node: The ConceptNode to generate a dataset for.

    Returns:
        A list of training data examples for the given node.
    """
    new_examples: List[Dict] = []
    if not start_node.problems_and_solutions:
      return []

    # Get ALL prerequisite descendants for this node. This is the key change.
    prereqs_to_process = self.graph.get_all_descendants(start_node.id)

    if not prereqs_to_process:
      print(
        f"  -> Node '{start_node.name}' has no prerequisites to analyze. Skipping."
      )
      return []

    max_depth = max(depth for _, depth in prereqs_to_process)

    print(
      f"\nProcessing node '{start_node.name}': "
      f"Analyzing {len(start_node.problems_and_solutions)} problems against {len(prereqs_to_process)} total prerequisites..."
    )

    log_time = time.monotonic()

    with ThreadPoolExecutor(
      max_workers=REPEATS_PER_PREREQ * len(prereqs_to_process)
    ) as executor:
      # A future for each (problem, prerequisite) pair
      future_to_pair = {
        executor.submit(
          self.teacher.analyze_error,
          problem,
          prereq,
          SOLUTIONS_PER_PREREQ * (max_depth - depth),
        ): (problem, prereq)
        for problem in start_node.problems_and_solutions
        for prereq, depth in prereqs_to_process
        for _ in range(REPEATS_PER_PREREQ)
      }

      for future in as_completed(future_to_pair):
        problem, prereq = future_to_pair[future]
        teacher_response = future.result()

        if teacher_response and teacher_response.is_valid_error:
          if time.monotonic() - log_time > 1:
            print(
              f"  -> [{len(new_examples)}] Valid error found for problem '{start_node.id}' with prerequisite '{prereq.id}'."
            )
            log_time = time.monotonic()
          for result in teacher_response.generated_solutions:
            new_examples.append(
              {
                "problem_example": problem.problem,
                "target_concept_id": start_node.id,
                "failure_concept_id": prereq.id,
                # Added back the step number for completeness
                "incorrect_step_number": result.step_number,
                "incorrect_solution": "\n".join(result.incorrect_solution),
              }
            )

    return new_examples
