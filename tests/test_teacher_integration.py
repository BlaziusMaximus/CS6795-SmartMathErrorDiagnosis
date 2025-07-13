"""Integration tests for the TeacherModel's analysis of a single problem."""

import pytest
from pathlib import Path
import re
from src.teacher import TeacherModel
from src.graph import KnowledgeGraph
from src.rate_limiter import ThreadSafeRateLimiter


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


@pytest.fixture(scope="module")
def rate_limiter() -> ThreadSafeRateLimiter:
  """Provides a shared rate limiter for integration tests."""
  return ThreadSafeRateLimiter(max_requests=999, period_seconds=60)


@pytest.mark.online
def test_single_problem_analysis_plausible_and_implausible(
  knowledge_graph: KnowledgeGraph, rate_limiter: ThreadSafeRateLimiter
):
  """
  Tests that the teacher can analyze a single problem against both a
  plausible and an implausible prerequisite error.
  """
  # Arrange
  teacher = TeacherModel(rate_limiter=rate_limiter)
  problem_node = knowledge_graph.get_node("934")  # Solving 2x2 Systems
  assert problem_node and problem_node.problems_and_solutions
  problem_to_test = problem_node.problems_and_solutions[0]

  # A plausible prerequisite for this problem
  plausible_prereq = knowledge_graph.get_node("864")  # Inverses of 2x2
  # An implausible prerequisite
  implausible_prereq = knowledge_graph.get_node("232")  # Transpose
  assert plausible_prereq and implausible_prereq

  # Act
  problem_analysis = teacher.analyze_single_problem(
    problem_and_solution=problem_to_test,
    failure_concepts=[plausible_prereq, implausible_prereq],
    solutions_to_generate=2,
  )

  # Assert
  assert problem_analysis is not None
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
