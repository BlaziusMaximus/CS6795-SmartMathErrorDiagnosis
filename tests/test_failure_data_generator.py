"""Tests for the FailureDataGenerator class."""

import pytest
from unittest.mock import MagicMock, patch
from src.failure_data_generator import (
  FailureDataGenerator,
  SOLUTIONS_BASE,
  SOLUTION_COUNT_DEPTH_DIVISOR,
)
from src.graph import KnowledgeGraph
from src.teacher import (
  SingleProblemAnalysis,
  PrerequisiteAnalysis,
  TeacherResponse,
  GeneratedErrorDetail,
)


@pytest.fixture
def mock_teacher():
  """Creates a mock TeacherModel."""
  return MagicMock()


@pytest.fixture
def knowledge_graph():
  """Loads the knowledge graph from the project's JSON file."""
  return KnowledgeGraph.load_from_json("knowledge_graph.json")


@patch(
  "src.failure_data_generator.FailureDataGenerator._process_node_portfolio"
)
def test_generate_error_dataset_traversal_loop(
  mock_process_portfolio, mock_teacher, knowledge_graph
):
  """
  Tests that the main traversal loop in generate_error_dataset works correctly.
  """
  # Arrange
  start_node_id = "1023"  # A node with prerequisites
  # Mock the return value of _process_node_portfolio to control the traversal
  mock_process_portfolio.return_value = (
    [],
    {"154"},
  )  # Return one node to visit

  # Act
  data_generator = FailureDataGenerator(knowledge_graph, mock_teacher)
  data_generator.generate_error_dataset(
    start_node_id=start_node_id, max_depth=2
  )

  # Assert
  # The processor should be called for the start node and its one prerequisite
  assert mock_process_portfolio.call_count == 2
  # Verify it was called with the correct nodes and depths
  call_args_list = mock_process_portfolio.call_args_list
  assert call_args_list[0].args[0].id == start_node_id
  assert call_args_list[0].args[1] == 1  # depth
  assert call_args_list[1].args[0].id == "154"
  assert call_args_list[1].args[1] == 2  # depth


def test_process_node_portfolio(mock_teacher, knowledge_graph):
  """
  Tests the logic for processing a single node's portfolio, checking if it
  correctly interprets the teacher's response and calls the teacher with
  the correct number of solutions to generate.
  """
  # Arrange
  current_node = knowledge_graph.get_node("934")  # Solving 2x2 Systems
  assert current_node is not None
  valid_prereq_id = "864"
  pruned_prereq_id = "1729"
  current_depth = 1  # Example depth

  # Mock the teacher's response
  mock_teacher.analyze_single_problem.return_value = SingleProblemAnalysis(
    problem_str="Problem 1",
    prerequisite_analyses=[
      PrerequisiteAnalysis(
        concept_id=valid_prereq_id,
        response=TeacherResponse(
          is_valid_error=True,
          reasoning="Valid.",
          generated_solutions=[
            GeneratedErrorDetail(step_number=1, incorrect_solution=["..."])
          ],
        ),
      ),
      PrerequisiteAnalysis(
        concept_id=pruned_prereq_id,
        response=TeacherResponse(is_valid_error=False, reasoning="Pruned."),
      ),
    ],
  )

  # Act
  data_generator = FailureDataGenerator(knowledge_graph, mock_teacher)
  new_examples, nodes_to_visit = data_generator._process_node_portfolio(
    current_node, current_depth=current_depth
  )

  # Assert
  # Since there are 5 problems in the test node, the teacher should be called 5 times.
  assert mock_teacher.analyze_single_problem.call_count == 5
  assert len(new_examples) == 5
  assert new_examples[0]["failure_concept_id"] == valid_prereq_id
  assert nodes_to_visit == {valid_prereq_id}
