from collections import deque
from typing import Deque, Dict, List, Set, Tuple
from .graph import KnowledgeGraph, ConceptNode
from .teacher import TeacherModel


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
    self.max_depth = 3

  def _process_node_portfolio(
    self, current_node: ConceptNode, visited: Set[str], current_depth: int
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

    prereq_nodes = (
      self.graph.get_node(p.concept_id)
      for p in current_node.prerequisites
      if p.concept_id not in visited
    )
    prerequisites_to_process = [node for node in prereq_nodes if node]

    if not prerequisites_to_process:
      return [], set()

    # Calculate how many solutions to generate based on depth
    # Depth 0 (direct prereqs) -> 10, Depth 1 -> 5, Depth 2 -> 2, etc.
    max_solutions = max(1, 10 // (2**current_depth))

    print(
      f"  -> Analyzing a portfolio of {len(current_node.problems_and_solutions)} problems "
      f"against {len(prerequisites_to_process)} prerequisites (gen up to {max_solutions} sols)..."
    )

    portfolio_response = self.teacher.analyze_problem_portfolio(
      problems_and_solutions=current_node.problems_and_solutions,
      failure_concepts=prerequisites_to_process,
      max_solutions_to_generate=max_solutions,
    )

    if not portfolio_response:
      print("    - Portfolio analysis failed or returned no data.")
      return [], set()

    for problem_analysis in portfolio_response.portfolio_analysis:
      for prereq_analysis in problem_analysis.prerequisite_analyses:
        if prereq_analysis.response.is_valid_error:
          prereq_id = prereq_analysis.concept_id
          nodes_to_visit_next.add(prereq_id)
          # Get the full prerequisite chain for this specific failure.
          prereq_chain = self.graph.get_all_descendants(prereq_id)
          prereq_chain_ids = [p.id for p in prereq_chain]

          for result in prereq_analysis.response.generated_solutions:
            new_examples.append(
              {
                "problem_example": problem_analysis.problem_str,
                "target_concept_id": current_node.id,
                "failure_concept_id": prereq_id,
                "prerequisite_chain": prereq_chain_ids,
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
    self.max_depth = max_depth
    full_dataset: List[Dict] = []
    queue: Deque[Tuple[str, int]] = deque([(start_node_id, 0)])
    visited: Set[str] = {start_node_id}

    start_node = self.graph.get_node(start_node_id)
    if start_node:
      print(
        "Starting dataset generation from node:"
        f"'{start_node.name}' with max depth: {self.max_depth}"
      )

    while queue:
      current_node_id, current_depth = queue.popleft()
      current_node = self.graph.get_node(current_node_id)

      if not current_node:
        print(f"Warning: Node '{current_node_id}' not found. Skipping.")
        continue

      if current_depth >= self.max_depth:
        print(
          f"Stopping at node '{current_node.name}' "
          f"(depth {current_depth}) due to max depth."
        )
        continue

      print(
        f"\nProcessing node '{current_node.name}' at depth {current_depth}..."
      )

      new_examples, nodes_to_visit_next = self._process_node_portfolio(
        current_node, visited, current_depth
      )
      full_dataset.extend(new_examples)

      for node_id in nodes_to_visit_next:
        if node_id not in visited:
          visited.add(node_id)
          queue.append((node_id, current_depth + 1))

    print("\nDataset generation complete.")
    return full_dataset
