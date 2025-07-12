"""Tests for the TeacherModel."""

import os
from unittest.mock import patch
import pytest
import json
from src.teacher import TeacherModel, TeacherResponse, GeneratedErrorDetail
from src.graph import ConceptNode


def test_teacher_model_initialization_success():
    """Tests that the TeacherModel can be initialized successfully."""
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}):
        with patch("google.genai.Client") as mock_client:
            teacher = TeacherModel()
            assert teacher is not None
            assert hasattr(teacher, 'client')
            mock_client.assert_called_once_with(api_key="test_key")


def test_teacher_model_initialization_raises_error_if_no_api_key():
    """Tests that TeacherModel raises a ValueError if the API key is missing."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="GEMINI_API_KEY environment variable not set."):
            TeacherModel()


def test_generate_synthetic_errors_success_case():
    """Tests the successful generation of synthetic errors."""
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}):
        with patch("google.genai.Client") as mock_client:
            mock_response = mock_client.return_value.models.generate_content
            mock_response.return_value.text = json.dumps({
                "is_valid_error": True,
                "reasoning": "This is a plausible error.",
                "generated_solutions": [
                    {
                        "step_number": 1,
                        "incorrect_solution": ["Solution 1"],
                    }
                ]
            })

            teacher = TeacherModel()
            result = teacher.generate_synthetic_errors(
                problem_example="2 + 2",
                correct_solution="4",
                failure_concept=ConceptNode(
                    id="1", name="Addition", description="...",
                    problem_and_solution={"problem": "2 + 2", "solution": "4"}
                ),
            )

            assert isinstance(result, TeacherResponse)
            assert result.is_valid_error is True
            assert result.generated_solutions
            assert isinstance(result.generated_solutions[0], GeneratedErrorDetail)


def test_generate_synthetic_errors_handles_malformed_json():
    """Tests that the method handles malformed JSON and returns None."""
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}):
        with patch("google.genai.Client") as mock_client:
            mock_response = mock_client.return_value.models.generate_content
            mock_response.return_value.text = "I am not JSON."

            teacher = TeacherModel()
            result = teacher.generate_synthetic_errors(
                problem_example="2 + 2",
                correct_solution="4",
                failure_concept=ConceptNode(
                    id="1", name="Addition", description="...",
                    problem_and_solution={"problem": "2 + 2", "solution": "4"}
                ),
            )

            assert result is None