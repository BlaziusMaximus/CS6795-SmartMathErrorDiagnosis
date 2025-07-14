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
    self,
    current_node: ConceptNode,
    prerequisites_to_process: List[ConceptNode],
    current_depth: int,
  ) -> Tuple[List[Dict], Set[str]]:
    """
    Analyzes a node's portfolio of problems against its prerequisites using a thread pool.
    """
    new_examples = []
    nodes_to_visit_next = set()
    if not current_node.problems_and_solutions or not prerequisites_to_process:
      return [], set()

    # Calculate how many solutions to generate based on depth
    max_solutions = max(
      1, SOLUTIONS_BASE // (SOLUTION_COUNT_DEPTH_DIVISOR**current_depth)
    )

    print(
      f"  -> Analyzing {len(current_node.problems_and_solutions)} problems "
      f"against {len(prerequisites_to_process)} prerequisites (gen up to {max_solutions} sols)..."
    )

    with ThreadPoolExecutor() as executor:
      futures = {
        executor.submit(
          self.teacher.analyze_single_problem,
          problem,
          prerequisites_to_process,
          max_solutions,
        ): problem
        for problem in current_node.problems_and_solutions
      }

      for future in as_completed(futures):
        problem_analysis = future.result()
        if not problem_analysis:
          print("    - Problem analysis failed or returned no data.")
          continue

        for prereq_analysis in problem_analysis.prerequisite_analyses:
          if not prereq_analysis.response.is_valid_error:
            print(
              f"    - Skipping invalid prerequisite {prereq_analysis.concept_id} for problem. Reasoning: {prereq_analysis.response.reasoning}"
            )
            continue

          nodes_to_visit_next.add(prereq_analysis.concept_id)
          for result in prereq_analysis.response.generated_solutions:
            new_examples.append(
              {
                "problem_example": problem_analysis.problem_str,
                "target_concept_id": current_node.id,
                "failure_concept_id": prereq_analysis.concept_id,
                "incorrect_solution": "\n".join(result.incorrect_solution),
                "incorrect_step_number": result.step_number,
              }
            )
    return new_examples, nodes_to_visit_next

  def generate_error_dataset(
    self, start_node_id: str, max_depth: int = 3
  ) -> list[dict]:
    """
    Generates a dataset by performing a breadth-first traversal.
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
          f"Stopping traversal at depth {current_depth}. Reached max depth."
        )
        continue

      current_node = self.graph.get_node(current_node_id)
      if not current_node:
        print(f"Warning: Node '{current_node_id}' not found. Skipping.")
        continue

      print(
        f"\nProcessing node '{current_node.id}':'{current_node.name}' at depth {current_depth}..."
      )

      # Get direct prerequisites and filter out any we have already visited
      direct_prereqs = self.graph.get_prerequisites(current_node.id)
      prereqs_to_visit = [p for p in direct_prereqs if p.id not in visited]

      if not prereqs_to_visit:
        print("  -> No new prerequisites to process for this node.")
        continue

      print(
        f"  -> Processing direct prerequisites {[p.id for p in prereqs_to_visit]}."
      )

      new_examples, nodes_to_visit_next = self._process_node_portfolio(
        current_node, prereqs_to_visit, current_depth
      )
      full_dataset.extend(new_examples)

      # Add the newly processed prerequisites to the queue and visited set
      for node_id in nodes_to_visit_next:
        if node_id not in visited:
          visited.add(node_id)
          queue.append((node_id, current_depth + 1))

    print("\nDataset generation complete.")
    return full_dataset
