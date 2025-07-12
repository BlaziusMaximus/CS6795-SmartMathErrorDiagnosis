"""Integration tests for the TeacherModel's portfolio analysis."""

import pytest
from pathlib import Path
import re
from src.teacher import TeacherModel
from src.graph import KnowledgeGraph


def normalize_string(s: str) -> str:
  """Removes all whitespace and newline characters from a string."""
  return re.sub(r"\s+", "", s)


# The project root directory, assuming tests are run from the root.
PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="module")
def knowledge_graph() -> KnowledgeGraph:
  """Loads the knowledge graph and makes it available to tests."""
  json_path = PROJECT_ROOT / "knowledge_graph.json"
  return KnowledgeGraph.load_from_json(json_path)


@pytest.mark.online
def test_portfolio_analysis_plausible_and_implausible(
  knowledge_graph: KnowledgeGraph,
):
  """
  Tests that the teacher can analyze a portfolio containing both a plausible
  and an implausible prerequisite error for a single problem.
  """
  # Arrange
  teacher = TeacherModel()
  problem_node = knowledge_graph.get_node("934")  # Solving 2x2 Systems
  assert problem_node and problem_node.problems_and_solutions
  problem_to_test = problem_node.problems_and_solutions[0]

  # A plausible prerequisite for this problem
  plausible_prereq = knowledge_graph.get_node("864")  # Inverses of 2x2
  # An implausible prerequisite
  implausible_prereq = knowledge_graph.get_node("232")  # Transpose
  assert plausible_prereq and implausible_prereq

  # Act
  portfolio_response = teacher.analyze_problem_portfolio(
    problems_and_solutions=[problem_to_test],
    failure_concepts=[plausible_prereq, implausible_prereq],
  )

  # Assert
  assert portfolio_response is not None
  assert len(portfolio_response.portfolio_analysis) == 1
  problem_analysis = portfolio_response.portfolio_analysis[0]
  assert normalize_string(problem_analysis.problem_str) == normalize_string(
    problem_to_test.problem
  )
  assert len(problem_analysis.prerequisite_analyses) == 2

  # Find the specific analyses from the response
  plausible_analysis = next(
    (
      p
      for p in problem_analysis.prerequisite_analyses
      if p.concept_id == plausible_prereq.id
    ),
    None,
  )
  implausible_analysis = next(
    (
      p
      for p in problem_analysis.prerequisite_analyses
      if p.concept_id == implausible_prereq.id
    ),
    None,
  )

  # Check the plausible case
  assert plausible_analysis is not None
  assert plausible_analysis.response.is_valid_error is True
  assert len(plausible_analysis.response.generated_solutions) > 0

  # Check the implausible case
  assert implausible_analysis is not None
  assert implausible_analysis.response.is_valid_error is False
  assert len(implausible_analysis.response.generated_solutions) == 0


@pytest.mark.online
def test_portfolio_analysis_multiple_problems(knowledge_graph: KnowledgeGraph):
  """
  Tests that the teacher can analyze a portfolio with multiple distinct
  problems against a single prerequisite.
  """
  # Arrange
  teacher = TeacherModel()

  # Problem 1: Solving a 2x2 system
  problem_node_1 = knowledge_graph.get_node("934")
  assert problem_node_1 and problem_node_1.problems_and_solutions
  problem_1 = problem_node_1.problems_and_solutions[0]

  # Problem 2: Finding a 3x3 determinant
  problem_node_2 = knowledge_graph.get_node("153")
  assert problem_node_2 and problem_node_2.problems_and_solutions
  problem_2 = problem_node_2.problems_and_solutions[0]

  # Prerequisite: Determinant of a 2x2 matrix (plausible for both)
  prereq = knowledge_graph.get_node("152")
  assert prereq

  # Act
  portfolio_response = teacher.analyze_problem_portfolio(
    problems_and_solutions=[problem_1, problem_2],
    failure_concepts=[prereq],
  )

  # Assert
  assert portfolio_response is not None
  assert len(portfolio_response.portfolio_analysis) == 2

  # Find the analysis for each problem
  analysis_1 = next(
    (
      p
      for p in portfolio_response.portfolio_analysis
      if normalize_string(p.problem_str) == normalize_string(problem_1.problem)
    ),
    None,
  )
  analysis_2 = next(
    (
      p
      for p in portfolio_response.portfolio_analysis
      if normalize_string(p.problem_str) == normalize_string(problem_2.problem)
    ),
    None,
  )

  # Check analysis for Problem 1
  assert analysis_1 is not None
  assert len(analysis_1.prerequisite_analyses) == 1
  prereq_analysis_1 = analysis_1.prerequisite_analyses[0]
  assert prereq_analysis_1.concept_id == prereq.id
  assert prereq_analysis_1.response.is_valid_error is True
  assert len(prereq_analysis_1.response.generated_solutions) > 0

  # Check analysis for Problem 2
  assert analysis_2 is not None
  assert len(analysis_2.prerequisite_analyses) == 1
  prereq_analysis_2 = analysis_2.prerequisite_analyses[0]
  assert prereq_analysis_2.concept_id == prereq.id
  assert prereq_analysis_2.response.is_valid_error is True
  assert len(prereq_analysis_2.response.generated_solutions) > 0
