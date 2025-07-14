from collections import deque
from typing import Deque, Dict, List, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from .graph import KnowledgeGraph
from .teacher import TeacherModel

# The base number of incorrect solutions to generate for direct prerequisites.
# This number is halved for each level deeper into the prerequisite graph.
SOLUTIONS_BASE = 50

# The divisor used to reduce the number of solutions generated at each
# increasing level of prerequisite depth.
SOLUTION_COUNT_DEPTH_DIVISOR = 2


class FailureDataGenerator:
  """
  Orchestrates the generation of a synthetic error dataset by intelligently
  traversing the knowledge graph.
  """

  def __init__(
    self, knowledge_graph: KnowledgeGraph, teacher_model: TeacherModel
  ):
    """
    Initializes the FailureDataGenerator.

    Args:
        knowledge_graph: An instance of the loaded knowledge graph.
        teacher_model: An instance of the TeacherModel for generating errors.
    """
    self.graph = knowledge_graph
    self.teacher = teacher_model

  def generate_error_dataset(
    self, start_node_id: str, max_depth: int = 3
  ) -> list[dict]:
    """
    Generates a dataset by performing a breadth-first traversal.

    For each node, it creates a batch of API calls to analyze every
    problem against every direct prerequisite, and processes them in parallel.
    """
    full_dataset: List[Dict] = []
    queue: Deque[Tuple[str, int]] = deque([(start_node_id, 0)])
    visited: Set[str] = {start_node_id}

    start_node = self.graph.get_node(start_node_id)
    if start_node:
      print(
        f"Starting dataset generation from node: '{start_node.name}' with max depth: {max_depth}"
      )

    while queue:
      current_node_id, current_depth = queue.popleft()

      if current_depth >= max_depth:
        print(
          f"Stopping traversal for this path at depth {current_depth}. Reached max depth."
        )
        continue

      current_node = self.graph.get_node(current_node_id)
      if not current_node or not current_node.problems_and_solutions:
        print(
          f"Warning: Node '{current_node_id}' not found or has no problems. Skipping."
        )
        continue

      print(
        f"\nProcessing node '{current_node_id}':'{current_node.name}' at depth {current_depth}..."
      )

      direct_prereqs = self.graph.get_prerequisites(current_node.id)
      prereqs_to_process = [p for p in direct_prereqs if p.id not in visited]

      if not prereqs_to_process:
        print("  -> No new prerequisites to process for this node.")
        continue

      print(
        f"  -> Found prerequisites to process: {[p.id for p in prereqs_to_process]}"
      )

      # --- NEW, SIMPLIFIED PARALLEL LOGIC ---
      nodes_to_visit_next = set()
      max_solutions = max(
        1, SOLUTIONS_BASE // (SOLUTION_COUNT_DEPTH_DIVISOR**current_depth)
      )

      with ThreadPoolExecutor() as executor:
        # Create a future for each (problem, prerequisite) pair
        future_to_pair = {
          executor.submit(
            self.teacher.analyze_error, problem, prereq, max_solutions
          ): (problem, prereq)
          for problem in current_node.problems_and_solutions
          for prereq in prereqs_to_process
        }

        for future in as_completed(future_to_pair):
          problem, prereq = future_to_pair[future]
          teacher_response = future.result()

          if teacher_response and teacher_response.is_valid_error:
            print(f"  -> Valid error found for problem/prereq: {prereq.name}")
            nodes_to_visit_next.add(prereq.id)
            for result in teacher_response.generated_solutions:
              full_dataset.append(
                {
                  "problem_example": problem.problem,
                  "target_concept_id": current_node.id,
                  "failure_concept_id": prereq.id,
                  "incorrect_solution": "\n".join(result.incorrect_solution),
                  "incorrect_step_number": result.step_number,
                }
              )
          else:
            reason = (
              teacher_response.reasoning if teacher_response else "No response"
            )
            print(
              f"  -> Skipping invalid prerequisite '{prereq.name}' for one problem. Reasoning: {reason}"
            )

      # Add all newly discovered valid nodes to the queue
      for node_id in nodes_to_visit_next:
        if node_id not in visited:
          visited.add(node_id)
          queue.append((node_id, current_depth + 1))

    print("\nDataset generation complete.")
    return full_dataset
