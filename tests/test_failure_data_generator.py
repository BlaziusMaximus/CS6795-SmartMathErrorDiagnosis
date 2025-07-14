"""Tests for the FailureDataGenerator class."""

import pytest
from unittest.mock import MagicMock
from src.failure_data_generator import FailureDataGenerator
from src.graph import KnowledgeGraph
from src.teacher import TeacherResponse, GeneratedErrorDetail


@pytest.fixture
def mock_teacher():
  """Creates a mock TeacherModel."""
  return MagicMock()


@pytest.fixture
def knowledge_graph():
  """Loads the knowledge graph from the project's JSON file."""
  return KnowledgeGraph.load_from_json("knowledge_graph.json")


def test_generate_error_dataset_with_pruning(mock_teacher, knowledge_graph):
  """
  Tests the full traversal and data generation logic, ensuring that the
  generator correctly uses the teacher's validation to prune paths and
  build the dataset.
  """
  # Arrange
  start_node_id = "1023"  # A node with multiple prerequisites

  # We will mock the teacher's response based on the prerequisite it receives.
  # This allows us to control the traversal path.
  def teacher_side_effect(problem, failure_concept, max_solutions):
    # When asked about this prerequisite, say it's VALID.
    if failure_concept.id == "154":
      return TeacherResponse(
        is_valid_error=True,
        reasoning="Plausible.",
        generated_solutions=[
          GeneratedErrorDetail(step_number=1, incorrect_solution=["..."])
        ],
      )
    # For all other prerequisites, say they are INVALID.
    else:
      return TeacherResponse(
        is_valid_error=False,
        reasoning="Implausible.",
        generated_solutions=[],
      )

  mock_teacher.analyze_error.side_effect = teacher_side_effect
  data_generator = FailureDataGenerator(knowledge_graph, mock_teacher)

  # Act
  # Run the full data generation with a depth of 2
  full_dataset = data_generator.generate_error_dataset(
    start_node_id=start_node_id, max_depth=2
  )

  # Assert
  # 1. The generator should have made calls for the direct prerequisites of "1023".
  #    Node 1023 has 5 problems and 3 prerequisites (154, 155, 934).
  #    Total calls at depth 0 = 5 * 3 = 15
  #
  # 2. Based on our mock, only prerequisite "154" was valid. So, the traversal
  #    should continue to node "154".
  #
  # 3. Node "154" has 5 problems and 3 prerequisites (153, 232, 864).
  #    Our mock will cause these to be pruned (is_valid_error=False).
  #    Total calls at depth 1 = 5 * 3 = 15
  #
  # 4. Total expected API calls = 15 (for node 1023) + 15 (for node 154) = 30
  assert mock_teacher.analyze_error.call_count == 30

  # 5. We should only have generated data for the single valid path.
  #    Node 1023 has 5 problems. For each problem, the prerequisite "154" was
  #    the only valid one, and our mock generates 1 solution for it.
  #    Total examples = 5 problems * 1 valid prereq * 1 solution = 5.
  assert len(full_dataset) == 5
  # All generated examples should point to the one valid failure concept.
  assert all(d["failure_concept_id"] == "154" for d in full_dataset)
