"""Unit tests for the TeacherModel."""

import os
import json
from unittest.mock import patch, MagicMock
import pytest
from src.teacher import TeacherModel, TeacherResponse
from src.graph import ConceptNode, ProblemSolutionPair


@pytest.fixture
def mock_env():
  """Fixture to set the GEMINI_API_KEY environment variable."""
  with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}):
    yield


@pytest.fixture
def mock_rate_limiter():
  """Fixture to create a mock rate limiter."""
  return MagicMock()


def test_teacher_model_initialization_success(mock_env, mock_rate_limiter):
  """Tests that the TeacherModel can be initialized successfully."""
  with patch("google.genai.Client") as mock_client:
    teacher = TeacherModel(rate_limiter=mock_rate_limiter)
    assert teacher is not None
    assert hasattr(teacher, "client")
    assert teacher.rate_limiter is mock_rate_limiter
    mock_client.assert_called_once_with(api_key="test_key")


def test_teacher_model_initialization_raises_error_if_no_api_key(
  mock_rate_limiter,
):
  """Tests that TeacherModel raises a ValueError if the API key is missing."""
  with patch.dict(os.environ, {}, clear=True):
    with pytest.raises(
      ValueError, match="GEMINI_API_KEY environment variable not set."
    ):
      TeacherModel(rate_limiter=mock_rate_limiter)


def test_analyze_error_success(mock_env, mock_rate_limiter):
  """Tests the successful analysis of a single problem and failure concept."""
  with patch("google.genai.Client") as mock_client:
    # Arrange
    # This mock response now matches the simpler TeacherResponse schema
    mock_api_response = {
      "is_valid_error": True,
      "reasoning": "This is a plausible error.",
      "generated_solutions": [
        {
          "step_number": 1,
          "incorrect_solution": ["Solution 1"],
        }
      ],
    }
    # The .text attribute is what the model_validate_json method will parse
    mock_client.return_value.models.generate_content.return_value.text = (
      json.dumps(mock_api_response)
    )

    teacher = TeacherModel(rate_limiter=mock_rate_limiter)
    problem = ProblemSolutionPair(problem="2 + 2", solution="4")
    failure_concept = ConceptNode(
      id="1", name="Addition", description="...", problems_and_solutions=[]
    )

    # Act
    result = teacher.analyze_error(
      problem, failure_concept, solutions_to_generate=5
    )

    # Assert
    mock_rate_limiter.acquire.assert_called_once()
    assert isinstance(result, TeacherResponse)
    assert result.is_valid_error is True
    assert len(result.generated_solutions) == 1


def test_analyze_error_handles_malformed_json(mock_env, mock_rate_limiter):
  """Tests that the method handles malformed JSON and returns None."""
  with patch("google.genai.Client") as mock_client:
    # Arrange
    mock_client.return_value.models.generate_content.return_value.text = (
      "This is not JSON"
    )

    teacher = TeacherModel(rate_limiter=mock_rate_limiter)
    problem = ProblemSolutionPair(problem="2 + 2", solution="4")
    failure_concept = ConceptNode(
      id="1", name="Addition", description="...", problems_and_solutions=[]
    )

    # Act
    result = teacher.analyze_error(
      problem, failure_concept, solutions_to_generate=5
    )

    # Assert
    assert result is None


def test_analyze_error_handles_validation_error(mock_env, mock_rate_limiter):
  """
  Tests that the method handles a valid JSON that does not match the schema.
  """
  with patch("google.genai.Client") as mock_client:
    # Arrange: Mock a response with a missing required field ("is_valid_error")
    mock_api_response = {"reasoning": "Missing a key field."}
    mock_client.return_value.models.generate_content.return_value.text = (
      json.dumps(mock_api_response)
    )

    teacher = TeacherModel(rate_limiter=mock_rate_limiter)
    problem = ProblemSolutionPair(problem="2 + 2", solution="4")
    failure_concept = ConceptNode(
      id="1", name="Addition", description="...", problems_and_solutions=[]
    )

    # Act
    result = teacher.analyze_error(
      problem, failure_concept, solutions_to_generate=5
    )

    # Assert
    assert result is None
