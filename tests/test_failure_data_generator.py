"""Tests for the FailureDataGenerator class."""

import pytest
from unittest.mock import MagicMock, patch
from src.failure_data_generator import FailureDataGenerator
from src.graph import KnowledgeGraph
from src.teacher import (
  PortfolioResponse,
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
  data_generator.max_depth = 2
  data_generator.generate_error_dataset(start_node_id=start_node_id)

  # Assert
  # The processor should be called for the start node and its one prerequisite
  assert mock_process_portfolio.call_count == 2
  # Verify it was called with the correct nodes and depths
  call_args_list = mock_process_portfolio.call_args_list
  assert call_args_list[0].args[0].id == start_node_id
  assert call_args_list[0].args[2] == 0  # depth
  assert call_args_list[1].args[0].id == "154"
  assert call_args_list[1].args[2] == 1  # depth


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
  mock_teacher.analyze_problem_portfolio.return_value = PortfolioResponse(
    portfolio_analysis=[
      SingleProblemAnalysis(
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
    ]
  )

  # Act
  data_generator = FailureDataGenerator(knowledge_graph, mock_teacher)
  new_examples, nodes_to_visit = data_generator._process_node_portfolio(
    current_node, visited=set(), current_depth=current_depth
  )

  # Assert
  assert len(new_examples) == 1
  assert new_examples[0]["failure_concept_id"] == valid_prereq_id
  assert nodes_to_visit == {valid_prereq_id}

  # Check that the teacher was called with the correct max_solutions
  expected_max_solutions = max(1, 10 // (2**current_depth))
  mock_teacher.analyze_problem_portfolio.assert_called_once()
  call_args = mock_teacher.analyze_problem_portfolio.call_args
  assert call_args.kwargs["max_solutions_to_generate"] == expected_max_solutions
