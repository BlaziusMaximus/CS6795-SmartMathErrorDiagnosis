"""Unit tests for the TeacherModel."""

import os
import json
from unittest.mock import patch
import pytest
from src.teacher import TeacherModel, PortfolioResponse
from src.graph import ConceptNode, ProblemSolutionPair


@pytest.fixture
def mock_env():
  """Fixture to set the GEMINI_API_KEY environment variable."""
  with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}):
    yield


def test_teacher_model_initialization_success(mock_env):
  """Tests that the TeacherModel can be initialized successfully."""
  with patch("google.genai.Client") as mock_client:
    teacher = TeacherModel()
    assert teacher is not None
    assert hasattr(teacher, "client")
    mock_client.assert_called_once_with(api_key="test_key")


def test_teacher_model_initialization_raises_error_if_no_api_key():
  """Tests that TeacherModel raises a ValueError if the API key is missing."""
  with patch.dict(os.environ, {}, clear=True):
    with pytest.raises(
      ValueError, match="GEMINI_API_KEY environment variable not set."
    ):
      TeacherModel()


def test_analyze_problem_portfolio_success(mock_env):
  """Tests the successful analysis of a problem portfolio."""
  with patch("google.genai.Client") as mock_client:
    # Arrange: Prepare a mock response that matches the PortfolioResponse schema
    mock_api_response = {
      "portfolio_analysis": [
        {
          "problem_str": "2 + 2",
          "prerequisite_analyses": [
            {
              "concept_id": "1",
              "response": {
                "is_valid_error": True,
                "reasoning": "This is a plausible error.",
                "generated_solutions": [
                  {
                    "step_number": 1,
                    "incorrect_solution": ["Solution 1"],
                  }
                ],
              },
            }
          ],
        }
      ]
    }
    mock_client.return_value.models.generate_content.return_value.text = (
      json.dumps(mock_api_response)
    )

    teacher = TeacherModel()
    problems = [ProblemSolutionPair(problem="2 + 2", solution="4")]
    concepts = [
      ConceptNode(
        id="1",
        name="Addition",
        description="...",
        problems_and_solutions=[],
      )
    ]

    # Act
    result = teacher.analyze_problem_portfolio(problems, concepts)

    # Assert
    assert isinstance(result, PortfolioResponse)
    assert len(result.portfolio_analysis) == 1
    analysis = result.portfolio_analysis[0]
    assert analysis.problem_str == "2 + 2"
    assert len(analysis.prerequisite_analyses) == 1
    prereq_analysis = analysis.prerequisite_analyses[0]
    assert prereq_analysis.concept_id == "1"
    assert prereq_analysis.response.is_valid_error is True


def test_analyze_problem_portfolio_handles_malformed_json(mock_env):
  """Tests that the method handles malformed JSON and returns None."""
  with patch("google.genai.Client") as mock_client:
    # Arrange: Mock an invalid JSON response
    mock_client.return_value.models.generate_content.return_value.text = (
      "This is not JSON"
    )

    teacher = TeacherModel()
    problems = [ProblemSolutionPair(problem="2 + 2", solution="4")]
    concepts = [
      ConceptNode(
        id="1",
        name="Addition",
        description="...",
        problems_and_solutions=[],
      )
    ]

    # Act
    result = teacher.analyze_problem_portfolio(problems, concepts)

    # Assert
    assert result is None


def test_analyze_problem_portfolio_handles_validation_error(mock_env):
  """
  Tests that the method handles a valid JSON that does not match the schema.
  """
  with patch("google.genai.Client") as mock_client:
    # Arrange: Mock a response with a missing required field
    mock_api_response = {
      "portfolio_analysis": [
        {
          "problem_str": "2 + 2",
          # "prerequisite_analyses" field is missing
        }
      ]
    }
    mock_client.return_value.models.generate_content.return_value.text = (
      json.dumps(mock_api_response)
    )

    teacher = TeacherModel()
    problems = [ProblemSolutionPair(problem="2 + 2", solution="4")]
    concepts = [
      ConceptNode(
        id="1",
        name="Addition",
        description="...",
        problems_and_solutions=[],
      )
    ]

    # Act
    result = teacher.analyze_problem_portfolio(problems, concepts)

    # Assert
    assert result is None