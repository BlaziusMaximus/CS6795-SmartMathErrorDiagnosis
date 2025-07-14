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


def test_generate_errors_for_node(mock_teacher, knowledge_graph):
  """
  Tests that the generator correctly analyzes a single node by:
  1. Finding all of its descendants.
  2. Calling the teacher model for each (problem, descendant) pair.
  3. Correctly processing the teacher's response to build a dataset.
  """
  # Arrange
  start_node = knowledge_graph.get_node("934")  # "Solving 2x2 Systems"
  assert start_node is not None

  # This node has 2 direct prerequisites (864, 1729) and a total of 13 descendants.
  all_descendants = knowledge_graph.get_all_descendants(start_node.id)
  assert len(all_descendants) == 13

  # This node has 5 problems in its portfolio.
  assert len(start_node.problems_and_solutions) == 5

  # Mock the teacher's response. We'll have it approve one specific
  # descendant ("Inverses of 2x2 Matrices") and reject all others.
  valid_failure_id = "864"

  def teacher_side_effect(problem, failure_concept, max_solutions):
    if failure_concept.id == valid_failure_id:
      return TeacherResponse(
        is_valid_error=True,
        reasoning="Plausible",
        generated_solutions=[
          GeneratedErrorDetail(step_number=1, incorrect_solution=["..."])
        ],
      )
    else:
      return TeacherResponse(is_valid_error=False, reasoning="Implausible")

  mock_teacher.analyze_error.side_effect = teacher_side_effect
  data_generator = FailureDataGenerator(knowledge_graph, mock_teacher)

  # Act
  # Run the data generation for just this single node.
  dataset = data_generator.generate_errors_for_node(start_node)

  # Assert
  # 1. The teacher should have been called for every combination of
  #    problem and descendant prerequisite.
  #    Total calls = 5 problems * 13 descendants = 65.
  assert mock_teacher.analyze_error.call_count == 65

  # 2. Our mock only returns a valid solution for the one `valid_failure_id`.
  #    Since there are 5 problems, we should get 5 training examples.
  assert len(dataset) == 5

  # 3. All generated examples should have the correct failure concept ID.
  assert all(d["failure_concept_id"] == valid_failure_id for d in dataset)
  # And they should all be for the correct target concept.
  assert all(d["target_concept_id"] == start_node.id for d in dataset)
