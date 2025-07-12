"""Integration tests for the TeacherModel."""

import os
import pytest
import google.genai as genai
import json
from src.teacher import TeacherModel
from src.graph import KnowledgeGraph


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
  raise ValueError("GEMINI_API_KEY environment variable not set.")


@pytest.mark.online
def test_teacher_generates_plausible_and_validatable_error():
  """Tests that the teacher can generate a plausible and validatable error."""
  # Setup
  teacher = TeacherModel()
  graph = KnowledgeGraph.load_from_json("knowledge_graph.json")
  failure_concept = graph.get_node("152")
  assert failure_concept is not None

  problem_example = (
    "Solve the system of linear equations: 2x + 3y = 7, x - y = 1."
  )
  correct_solution = (
    "Step 1: Represent the system as a matrix equation AX=B: A = [[2, 3], [1, -1]], X = [[x], [y]], B = [[7], [1]].\n"
    "Step 2: Calculate the determinant of A: det(A) = (2 * -1) - (3 * 1) = -2 - 3 = -5.\n"
    "Step 3: Calculate the inverse of A: A-inverse = (1/det(A)) * [[-1, -3], [-1, 2]] = (1/-5) * [[-1, -3], [-1, 2]] = [[1/5, 3/5], [1/5, -2/5]].\n"
    "Step 4: Solve for X: X = A-inverse * B = [[1/5, 3/5], [1/5, -2/5]] * [[7], [1]] = [[(1/5)*7 + (3/5)*1], [(1/5)*7 + (-2/5)*1]] = [[7/5 + 3/5], [7/5 - 2/5]] = [[10/5], [5/5]] = [[2], [1]].\n"
    "Step 5: Extract x and y: x = 2, y = 1."
  )

  # Generate a synthetic error
  teacher_response = teacher.generate_synthetic_errors(
    problem_example=problem_example,
    correct_solution=correct_solution,
    failure_concept=failure_concept,
  )

  assert teacher_response is not None
  assert teacher_response.is_valid_error is True
  assert teacher_response.generated_solutions
  incorrect_solution = "\n".join(
    teacher_response.generated_solutions[0].incorrect_solution
  )

  # Validate the generated error with a second API call
  validator_client = genai.Client(api_key=GEMINI_API_KEY)
  validation_prompt = f"""
    Analyze the provided incorrect solution for the problem '{problem_example}'.

    Incorrect solution:
    {incorrect_solution}

    Respond with a JSON object with the following schema: {{\"error_was_made\": <true_or_false>, \"identified_error_concept\": \"<name_of_concept>\"}}.
    """

  response = validator_client.models.generate_content(
    model="gemini-2.5-flash",
    contents=validation_prompt,
    config={"response_mime_type": "application/json"},
  )

  assert response is not None
  assert response.text is not None
  validation_json = json.loads(response.text)

  assert validation_json["error_was_made"] is True
  assert "Determinant" in validation_json["identified_error_concept"]


@pytest.mark.online
def test_teacher_rejects_implausible_error():
  """Tests that the teacher correctly rejects an implausible error."""
  # Setup
  teacher = TeacherModel()
  graph = KnowledgeGraph.load_from_json("knowledge_graph.json")
  # Use an implausible failure concept, e.g., "The Transpose of a Matrix" (ID: 232)
  implausible_failure_concept = graph.get_node("232")
  assert implausible_failure_concept is not None

  problem_example = "Calculate the determinant of [[5, -2], [3, 1]]."
  correct_solution = (
    "Step 1: Identify the elements of the 2x2 matrix [[a, b], [c, d]]. For [[5, -2], [3, 1]], a=5, b=-2, c=3, d=1.\n"
    "Step 2: Apply the determinant formula: det(A) = ad - bc.\n"
    "Step 3: Substitute the values: det(A) = (5 * 1) - (-2 * 3).\n"
    "Step 4: Perform multiplication: (5 * 1) = 5 and (-2 * 3) = -6.\n"
    "Step 5: Perform subtraction: 5 - (-6) = 5 + 6 = 11.\n"
    "Step 6: The determinant is 11."
  )

  # Generate a synthetic error with an implausible concept
  teacher_response = teacher.generate_synthetic_errors(
    problem_example=problem_example,
    correct_solution=correct_solution,
    failure_concept=implausible_failure_concept,
  )

  assert teacher_response is not None
  assert teacher_response.is_valid_error is False
  assert not teacher_response.generated_solutions
  assert (
    "transpose" in teacher_response.reasoning.lower()
    or "not relevant" in teacher_response.reasoning.lower()
  )


@pytest.mark.online
def test_teacher_handles_deeper_prerequisite_error():
  """Tests that the teacher can handle a deeper prerequisite error."""
  # Setup
  teacher = TeacherModel()
  graph = KnowledgeGraph.load_from_json("knowledge_graph.json")
  # Deeper failure concept: "Introduction to Matrices" (ID: 861)
  deeper_failure_concept = graph.get_node("861")
  assert deeper_failure_concept is not None

  problem_example = "Consider the system of linear equations: 2x+y-z=1, x+2y+z=8, x-y+2z=7. Given the inverse of the coefficient matrix, find the value of y."
  correct_solution = (
    "Step 1: Represent the system as AX=B. A = [[2, 1, -1], [1, 2, 1], [1, -1, 2]], X = [[x], [y], [z]], B = [[1], [8], [7]].\n"
    "Step 2: Given A-inverse, calculate X = A-inverse * B. For example, if A-inverse = [[a,b,c],[d,e,f],[g,h,i]], then X = [[a*1+b*8+c*7],[d*1+e*8+f*7],[g*1+h*8+i*7]].\n"
    "Step 3: Extract the value of y from the resulting X vector."
  )

  # Generate a synthetic error
  teacher_response = teacher.generate_synthetic_errors(
    problem_example=problem_example,
    correct_solution=correct_solution,
    failure_concept=deeper_failure_concept,
  )

  assert teacher_response is not None
  assert teacher_response.is_valid_error is True
  assert teacher_response.generated_solutions
  incorrect_solution = "\n".join(
    teacher_response.generated_solutions[0].incorrect_solution
  )

  # Validate the generated error with a second API call
  validator_client = genai.Client(api_key=GEMINI_API_KEY)
  validation_prompt = f"""
    Analyze the provided incorrect solution for the problem '{problem_example}'.

    Incorrect solution:
    {incorrect_solution}

    Respond with a JSON object with the following schema: {{\"error_was_made\": <true_or_false>, \"identified_error_concept\": \"<name_of_concept>\"}}.
    """

  response = validator_client.models.generate_content(
    model="gemini-2.5-flash",
    contents=validation_prompt,
    config={"response_mime_type": "application/json"},
  )

  assert response is not None
  assert response.text is not None
  validation_json = json.loads(response.text)

  assert validation_json["error_was_made"] is True
  assert "matrix" in validation_json["identified_error_concept"].lower()
