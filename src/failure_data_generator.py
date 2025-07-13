from collections import deque
from typing import Deque, Dict, List, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from .graph import KnowledgeGraph, ConceptNode
from .teacher import TeacherModel

# The base number of incorrect solutions to generate for direct prerequisites.
# This number is halved for each level deeper into the prerequisite graph.
SOLUTIONS_BASE = 70

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

  def _process_node_portfolio(
    self, current_node: ConceptNode, current_depth: int
  ) -> Tuple[List[Dict], Set[str]]:
    """
    Analyzes a node's portfolio of problems against its prerequisites.

    Args:
        current_node: The node whose problems are to be analyzed.
        visited: A set of visited node IDs.
        current_depth: The current depth in the graph traversal.

    Returns:
        A tuple containing:
        - A list of newly generated training examples.
        - A set of prerequisite IDs that are valid for the next traversal level.
    """
    new_examples = []
    nodes_to_visit_next = set()

    if not current_node.problems_and_solutions:
      return [], set()

    prerequisites_to_process = self.graph.get_prerequisites(
      current_node.id, depth=current_depth
    )

    if not prerequisites_to_process:
      return [], set()

    # Calculate how many solutions to generate based on depth
    # Depth 0 (direct prereqs) -> 10, Depth 1 -> 5, Depth 2 -> 2, etc.
    num_solutions = max(
      1,
      SOLUTIONS_BASE // (SOLUTION_COUNT_DEPTH_DIVISOR ** (current_depth - 1)),
    )

    with ThreadPoolExecutor() as executor:
      futures = {
        executor.submit(
          self.teacher.analyze_single_problem,
          problem,
          prerequisites_to_process,
          num_solutions,
        ): problem
        for problem in current_node.problems_and_solutions
      }

      for future in as_completed(futures):
        problem_analysis = future.result()
        if not problem_analysis:
          print("    - Problem analysis failed or returned no data.")
          continue

        for prereq_analysis in problem_analysis.prerequisite_analyses:
          if prereq_analysis.response.is_valid_error:
            prereq_id = prereq_analysis.concept_id
            nodes_to_visit_next.add(prereq_id)

            for result in prereq_analysis.response.generated_solutions:
              new_examples.append(
                {
                  "problem_example": problem_analysis.problem_str,
                  "target_concept_id": current_node.id,
                  "failure_concept_id": prereq_id,
                  "incorrect_solution": "\n".join(result.incorrect_solution),
                  "incorrect_step_number": result.step_number,
                }
              )
    return new_examples, nodes_to_visit_next

  def generate_error_dataset(
    self, start_node_id: str, max_depth: int = 3
  ) -> list[dict]:
    """
    Generates a dataset of synthetic errors by traversing the knowledge graph.

    This method performs a breadth-first traversal starting from the
    `start_node_id`. At each node, it analyzes all of that node's problems
    against its direct prerequisites in a single batch API call.

    Args:
        start_node_id: The ID of the concept node to start traversal from.
        max_depth: The maximum depth to traverse down the prerequisite tree.
    """
    full_dataset: List[Dict] = []
    # Initialize the queue for BFS traversal
    queue: Deque[Tuple[str, int]] = deque([(start_node_id, 1)])
    # Set to track visited nodes
    visited: Set[str] = {start_node_id}

    start_node = self.graph.get_node(start_node_id)
    if start_node:
      print(
        "Starting dataset generation from node:"
        f"'{start_node.name}' with max depth: {max_depth}"
      )

    while queue:
      current_node_id, current_depth = queue.popleft()
      current_node = self.graph.get_node(current_node_id)

      if not current_node:
        print(f"Warning: Node '{current_node_id}' not found. Skipping.")
        continue

      if current_depth > max_depth:
        print(
          f"Stopping at node '{current_node.name}' "
          f"(depth {current_depth}) due to max depth."
        )
        continue

      print(
        f"\nProcessing node '{current_node.name}' at depth {current_depth}..."
      )

      new_examples, nodes_to_visit_next = self._process_node_portfolio(
        current_node, current_depth
      )
      full_dataset.extend(new_examples)

      for node_id in nodes_to_visit_next:
        if node_id not in visited:
          visited.add(node_id)
          queue.append((node_id, current_depth + 1))

    print("\nDataset generation complete.")
    return full_dataset
