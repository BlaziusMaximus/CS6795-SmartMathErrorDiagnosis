"""Tests for the knowledge graph data models."""

from pathlib import Path
from src.graph import KnowledgeGraph, ConceptNode

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
  assert len(graph.nodes) == 3

  # Check for a specific node
  multiply_node = graph.nodes.get("multiply_2d_by_1d")
  assert multiply_node is not None
  assert isinstance(multiply_node, ConceptNode)
  assert (
    multiply_node.name == "Multiplying a Two-Digit Number by a One-Digit Number"
  )

  # Check prerequisites
  assert len(multiply_node.prerequisites) == 2
  prereq_ids = {p.concept_id for p in multiply_node.prerequisites}
  assert "multiply_1d_numbers" in prereq_ids
  assert "add_1d_to_2d" in prereq_ids
