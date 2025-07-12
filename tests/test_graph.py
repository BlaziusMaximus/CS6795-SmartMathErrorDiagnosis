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
