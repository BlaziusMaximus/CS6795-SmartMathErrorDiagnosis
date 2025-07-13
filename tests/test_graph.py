"""Tests for the knowledge graph data models."""

import pytest
from pathlib import Path
from src.graph import KnowledgeGraph

# The project root directory, assuming tests are run from the root.
PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="module")
def knowledge_graph() -> KnowledgeGraph:
  """Loads the knowledge graph and makes it available to tests."""
  json_path = PROJECT_ROOT / "knowledge_graph.json"
  return KnowledgeGraph.load_from_json(json_path)


def test_load_from_json():
  """Tests that the knowledge graph can be loaded correctly from a JSON file."""
  # Path to the test data
  json_path = PROJECT_ROOT / "knowledge_graph.json"

  # Load the graph
  graph = KnowledgeGraph.load_from_json(json_path)

  # Assertions
  assert graph is not None
  assert isinstance(graph, KnowledgeGraph)
  assert len(graph.nodes) == 18


def test_get_node(knowledge_graph: KnowledgeGraph):
  """Tests retrieving a node by its ID."""
  # Get a node
  node = knowledge_graph.get_node("154")

  # Assertions
  assert node is not None
  assert node.id == "154"
  assert (
    node.name
    == "Calculating the Inverse of a 3x3 Matrix Using the Cofactor Method"
  )


def test_get_prerequisites(knowledge_graph: KnowledgeGraph):
  """Tests retrieving the prerequisites for a given node."""
  # Get prerequisites
  prerequisites = knowledge_graph.get_prerequisites("1023")

  # Assertions
  assert prerequisites is not None
  assert len(prerequisites) == 3
  prereq_ids = {p.id for p in prerequisites}
  assert "154" in prereq_ids
  assert "155" in prereq_ids
  assert "934" in prereq_ids


def test_get_prerequisites_at_depth(knowledge_graph: KnowledgeGraph):
  """Tests retrieving prerequisites at a depth greater than 1."""
  # The node "Solving 4x4 Systems" (1023) has prerequisites at multiple depths.
  # A known prerequisite path is:
  # 1023 -> 934 ("Solving 2x2 Systems") -> 864 ("Inverse of 2x2") -> 152 ("Determinant of 2x2")
  start_node_id = "1023"

  # Get prerequisites at depth 2
  prerequisites_depth_2 = knowledge_graph.get_prerequisites(
    start_node_id, depth=2
  )
  prereq_ids_depth_2 = {p.id for p in prerequisites_depth_2}

  # Assertions for depth 2
  # We expect to find the prerequisites of this node's direct prerequisites
  # ("154", "155", "934"). From other tests, we know "864" is a prereq of "934".
  assert "864" in prereq_ids_depth_2

  # Get prerequisites at depth 3
  prerequisites_depth_3 = knowledge_graph.get_prerequisites(
    start_node_id, depth=3
  )
  prereq_ids_depth_3 = {p.id for p in prerequisites_depth_3}

  # From other tests, we know "152" is a prereq of "864".
  assert "152" in prereq_ids_depth_3

  # Test that the function returns unique nodes, even if reached via multiple paths.
  assert len(prerequisites_depth_2) == len(prereq_ids_depth_2)


def test_get_prerequisites_with_invalid_depth(knowledge_graph: KnowledgeGraph):
  """Tests that get_prerequisites raises an error for invalid depth."""
  with pytest.raises(AssertionError, match="Depth must be at least 1"):
    knowledge_graph.get_prerequisites("1023", depth=0)


def test_node_with_problems_and_solutions(knowledge_graph: KnowledgeGraph):
  """Tests that a node with problems and solutions is loaded correctly."""
  # Get a node with a problem
  node = knowledge_graph.get_node("1023")

  # Assertions
  assert node is not None
  assert node.problems_and_solutions is not None
  assert isinstance(node.problems_and_solutions, list)
  assert len(node.problems_and_solutions) > 0
  first_problem = node.problems_and_solutions[0]
  assert isinstance(first_problem.problem, str)
  assert isinstance(first_problem.solution, str)
  assert "Solve the following system" in first_problem.problem


def test_node_without_prerequisites(knowledge_graph: KnowledgeGraph):
  """Tests a node that has no prerequisites."""
  # Get a node with no prerequisites
  node = knowledge_graph.get_node("861")

  # Assertions
  assert node is not None
  assert node.prerequisites == []
  prerequisites = knowledge_graph.get_prerequisites("861")
  assert prerequisites == []


def test_get_all_descendants(knowledge_graph: KnowledgeGraph):
  """
  Tests that all unique descendants for a node are retrieved correctly.
  """
  # The node "Solving 2x2 Systems" (934) has a known, multi-level tree.
  # 934 -> 864 -> 152 -> 1166 -> 1167 -> 861 (and other branches)
  start_node_id = "934"
  descendants = knowledge_graph.get_all_descendants(start_node_id)

  # Assertions
  assert descendants is not None
  descendant_ids = {d.id for d in descendants}

  # Check for specific nodes at different depths
  assert "864" in descendant_ids  # Direct prerequisite (depth 1)
  assert "152" in descendant_ids  # Prereq of 864 (depth 2)
  assert "1166" in descendant_ids  # Prereq of 152 (depth 3)
  assert "861" in descendant_ids  # Deepest node

  # Check that the start node itself is not in the list
  assert start_node_id not in descendant_ids

  # Check that there are no duplicates
  assert len(descendants) == len(descendant_ids)
