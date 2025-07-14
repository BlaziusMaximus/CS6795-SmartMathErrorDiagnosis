"""Integration tests for the TeacherModel's analysis."""

import pytest
from pathlib import Path
from src.teacher import TeacherModel
from src.graph import KnowledgeGraph
from src.rate_limiter import ThreadSafeRateLimiter


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
def test_analyze_error_plausible_case(
  knowledge_graph: KnowledgeGraph, rate_limiter: ThreadSafeRateLimiter
):
  """
  Tests that the teacher correctly identifies a PLAUSIBLE prerequisite error
  and generates solutions.
  """
  # Arrange
  teacher = TeacherModel(rate_limiter=rate_limiter)
  problem_node = knowledge_graph.get_node("934")  # Solving 2x2 Systems
  assert problem_node and problem_node.problems_and_solutions
  problem_to_test = problem_node.problems_and_solutions[0]

  # A plausible prerequisite for this problem
  plausible_prereq = knowledge_graph.get_node("864")  # Inverses of 2x2
  assert plausible_prereq

  # Act
  response = teacher.analyze_error(
    problem_and_solution=problem_to_test,
    failure_concept=plausible_prereq,
    solutions_to_generate=2,
  )

  # Assert
  assert response is not None
  assert response.is_valid_error is True
  assert len(response.generated_solutions) > 0


@pytest.mark.online
def test_analyze_error_implausible_case(
  knowledge_graph: KnowledgeGraph, rate_limiter: ThreadSafeRateLimiter
):
  """
  Tests that the teacher correctly identifies an IMPLAUSIBLE prerequisite
  error and does NOT generate solutions.
  """
  # Arrange
  teacher = TeacherModel(rate_limiter=rate_limiter)
  problem_node = knowledge_graph.get_node("861")  # Solving 2x2 Systems
  assert problem_node and problem_node.problems_and_solutions
  problem_to_test = problem_node.problems_and_solutions[0]

  # An implausible prerequisite for this problem
  implausible_prereq = knowledge_graph.get_node("1023")  # Transpose
  assert implausible_prereq

  # Act
  response = teacher.analyze_error(
    problem_and_solution=problem_to_test,
    failure_concept=implausible_prereq,
    solutions_to_generate=2,
  )

  # Assert
  assert response is not None
  assert response.is_valid_error is False
  assert len(response.generated_solutions) == 0
