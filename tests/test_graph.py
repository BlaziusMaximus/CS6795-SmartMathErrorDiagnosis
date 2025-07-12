"""Tests for the knowledge graph data models."""

from pathlib import Path
from src.graph import KnowledgeGraph

# The project root directory, assuming tests are run from the root.
PROJECT_ROOT = Path(__file__).parent.parent


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


def test_get_node():
  """Tests retrieving a node by its ID."""
  # Path to the test data
  json_path = PROJECT_ROOT / "knowledge_graph.json"

  # Load the graph
  graph = KnowledgeGraph.load_from_json(json_path)

  # Get a node
  node = graph.get_node("154")

  # Assertions
  assert node is not None
  assert node.id == "154"
  assert (
    node.name
    == "Calculating the Inverse of a 3x3 Matrix Using the Cofactor Method"
  )


def test_get_prerequisites():
  """Tests retrieving the prerequisites for a given node."""
  # Path to the test data
  json_path = PROJECT_ROOT / "knowledge_graph.json"

  # Load the graph
  graph = KnowledgeGraph.load_from_json(json_path)

  # Get prerequisites
  prerequisites = graph.get_prerequisites("1023")

  # Assertions
  assert prerequisites is not None
  assert len(prerequisites) == 3
  prereq_ids = {p.id for p in prerequisites}
  assert "154" in prereq_ids
  assert "155" in prereq_ids
  assert "934" in prereq_ids
