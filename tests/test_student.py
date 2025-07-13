"""Tests for the StudentModel and ErrorDiagnosisDataset."""

import pytest
from unittest.mock import MagicMock
from src.student import StudentModel, ErrorDiagnosisDataset
from src.graph import KnowledgeGraph
from transformers import AutoTokenizer


@pytest.fixture
def mock_knowledge_graph():
  """Creates a mock KnowledgeGraph."""
  return MagicMock(spec=KnowledgeGraph)


@pytest.fixture
def tokenizer():
  """Provides a tokenizer for the tests."""
  return AutoTokenizer.from_pretrained("distilbert-base-uncased")


def test_error_diagnosis_dataset(mock_knowledge_graph, tokenizer):
  """
  Tests that the ErrorDiagnosisDataset correctly formats the input text by
  combining the problem, prerequisite chain, and solution.
  """
  # Arrange
  dataset_list = [
    {
      "problem_example": "What is 2+2?",
      "target_concept_id": "1",
      "failure_concept_id": "2",
      "incorrect_solution": "The answer is 5.",
    }
  ]
  label_map = {"2": 0}
  # Mock the return value of get_all_descendants
  mock_descendants = [MagicMock(), MagicMock()]
  mock_descendants[0].name = "Addition"
  mock_descendants[1].name = "Counting"
  mock_knowledge_graph.get_all_descendants.return_value = mock_descendants

  # Act
  dataset = ErrorDiagnosisDataset(
    dataset_list, tokenizer, label_map, mock_knowledge_graph
  )
  item = dataset[0]
  decoded_text = tokenizer.decode(item["input_ids"].squeeze())

  # Assert
  assert "problem : what is 2 + 2?" in decoded_text.lower()
  assert "relevant concepts : addition, counting" in decoded_text.lower()
  assert "solution : the answer is 5." in decoded_text.lower()
  assert item["labels"].item() == 0


def test_student_model_initialization():
  """
  Tests that the StudentModel can be initialized correctly.
  """
  # Arrange
  num_labels = 10
  model_name = "distilbert-base-uncased"

  # Act
  model = StudentModel(num_labels=num_labels, model_name=model_name)

  # Assert
  assert model is not None
  assert model.bert.config.num_labels == num_labels
