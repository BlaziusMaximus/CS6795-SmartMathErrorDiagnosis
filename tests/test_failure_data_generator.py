"""Tests for the FailureDataGenerator class."""

import pytest
from unittest.mock import MagicMock
from src.failure_data_generator import (
  FailureDataGenerator,
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


def test_generate_error_dataset_traversal_logic(mock_teacher, knowledge_graph):
  """
  Tests that the main traversal loop correctly explores the graph based on
  the teacher's validation response.
  """
  # Arrange
  start_node_id = "1023"  # Top-level node
  direct_prereq_id = "154"
  deep_prereq_id = "153"

  # Mock the teacher to control which paths are valid
  def teacher_side_effect(problem, failure_concepts, max_solutions):
    # When analyzing node 1023, only prereq 154 is valid
    if (
      problem.problem
      == knowledge_graph.get_node("1023").problems_and_solutions[0].problem
    ):
      return SingleProblemAnalysis(
        problem_str=problem.problem,
        prerequisite_analyses=[
          PrerequisiteAnalysis(
            concept_id=direct_prereq_id,
            response=TeacherResponse(
              is_valid_error=True, reasoning="", generated_solutions=[]
            ),
          ),
          PrerequisiteAnalysis(
            concept_id="155",
            response=TeacherResponse(is_valid_error=False, reasoning=""),
          ),
          PrerequisiteAnalysis(
            concept_id="934",
            response=TeacherResponse(is_valid_error=False, reasoning=""),
          ),
        ],
      )
    # When analyzing node 154, only prereq 153 is valid
    elif (
      problem.problem
      == knowledge_graph.get_node("154").problems_and_solutions[0].problem
    ):
      return SingleProblemAnalysis(
        problem_str=problem.problem,
        prerequisite_analyses=[
          PrerequisiteAnalysis(
            concept_id=deep_prereq_id,
            response=TeacherResponse(
              is_valid_error=True, reasoning="", generated_solutions=[]
            ),
          ),
        ],
      )
    return None  # Default case

  mock_teacher.analyze_single_problem.side_effect = teacher_side_effect

  # Act
  data_generator = FailureDataGenerator(knowledge_graph, mock_teacher)
  data_generator.generate_error_dataset(
    start_node_id=start_node_id, max_depth=2
  )

  # Assert
  # We expect analyze_single_problem to be called for:
  # 1. The 5 problems in node 1023
  # 2. The 5 problems in node 154 (the only valid path)
  # Total calls = 10
  assert mock_teacher.analyze_single_problem.call_count == 10


def test_process_node_portfolio(mock_teacher, knowledge_graph):
  """
  Tests the logic for processing a single node's portfolio, checking if it
  correctly interprets the teacher's response.
  """
  # Arrange
  current_node = knowledge_graph.get_node("934")  # Solving 2x2 Systems
  assert current_node and current_node.problems_and_solutions

  # The direct prerequisites of the current node
  prereqs_to_process = knowledge_graph.get_prerequisites(current_node.id)
  assert len(prereqs_to_process) == 2  # Should be 864 and 1729

  valid_prereq_id = "864"
  pruned_prereq_id = "1729"
  current_depth = 0

  # CORRECTED MOCK: This now correctly simulates the response for ONE problem analysis.
  # The TeacherModel will be called multiple times, once for each problem,
  # and we can use the same mock response for all of them in this test.
  mock_teacher.analyze_single_problem.return_value = SingleProblemAnalysis(
    problem_str="A sample problem.",  # This will be consistent for all returns
    prerequisite_analyses=[
      PrerequisiteAnalysis(
        concept_id=valid_prereq_id,
        response=TeacherResponse(
          is_valid_error=True,
          reasoning="This one is valid.",
          generated_solutions=[
            GeneratedErrorDetail(step_number=2, incorrect_solution=["..."])
          ],
        ),
      ),
      PrerequisiteAnalysis(
        concept_id=pruned_prereq_id,
        response=TeacherResponse(
          is_valid_error=False, reasoning="This one is not valid."
        ),
      ),
    ],
  )

  # Act
  data_generator = FailureDataGenerator(knowledge_graph, mock_teacher)
  new_examples, nodes_to_visit = data_generator._process_node_portfolio(
    current_node, prereqs_to_process, current_depth=current_depth
  )

  # Assert
  # The teacher's method is called once for each of the 5 problems in the portfolio.
  assert mock_teacher.analyze_single_problem.call_count == 5

  # Each of the 5 calls finds one valid prerequisite, which has one generated solution.
  # So we expect 5 * 1 = 5 total examples.
  assert len(new_examples) == 5
  assert new_examples[0]["failure_concept_id"] == valid_prereq_id

  assert nodes_to_visit == {valid_prereq_id}
