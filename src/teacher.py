import os
import json
from google import genai
from pydantic import BaseModel, Field, ValidationError
from typing import List
from .graph import ConceptNode, ProblemSolutionPair
from .rate_limiter import ThreadSafeRateLimiter


class GeneratedErrorDetail(BaseModel):
  """Details of a generated synthetic solution."""

  step_number: int
  incorrect_solution: List[str]


class TeacherResponse(BaseModel):
  """The full response from the teacher model, including validation."""

  is_valid_error: bool
  reasoning: str
  generated_solutions: List[GeneratedErrorDetail] = Field(default_factory=list)


class TeacherModel:
  """A model representing the 'Teacher' component for generating synthetic errors."""

  def __init__(self, rate_limiter: ThreadSafeRateLimiter):
    """
    Initializes the TeacherModel and configures the Gemini API client.

    Args:
        rate_limiter: An instance of a thread-safe rate limiter.
    """
    if not os.getenv("GEMINI_API_KEY"):
      raise ValueError("GEMINI_API_KEY environment variable not set.")

    self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    self.rate_limiter = rate_limiter

  def analyze_error(
    self,
    problem_and_solution: ProblemSolutionPair,
    failure_concept: ConceptNode,
    solutions_to_generate: int,
  ) -> TeacherResponse | None:
    """
    Analyzes a single problem and a single potential prerequisite failure.

    This is a more focused method designed to produce higher-quality,
    more reliable validation from the LLM.

    Args:
        problem_and_solution: The problem-solution pair to analyze.
        failure_concept: The single prerequisite concept to check for errors.
        solutions_to_generate: The max number of incorrect solutions to generate.

    Returns:
        A TeacherResponse object containing the analysis, or None if the
        API response is invalid.
    """
    system_instruction = (
      "You are an expert in math pedagogy. Your task is to analyze a math problem and a "
      "potential prerequisite error. Your primary goal is to determine if a student's "
      "misunderstanding of the prerequisite concept could plausibly lead to an incorrect "
      "solution for the main problem. Think like a real tutor: consider how concepts are "
      "truly connected, even if not explicitly stated. If the connection is plausible, "
      "generate examples of the error."
    )

    prompt = f"""
        **Problem Context:**
        Problem: "{problem_and_solution.problem}"
        Correct Solution Steps: "{problem_and_solution.solution}"

        **Potential Prerequisite Error to Analyze:**
        Concept Name: "{failure_concept.name}"
        Description: "{failure_concept.description}"

        **Your Tasks:**
        1.  **Validation:** Based on the problem and its solution, is it plausible that a misunderstanding of the prerequisite concept '{failure_concept.name}' could cause an error? The connection can be direct or indirect.
        2.  **Generation:** If and only if the error is plausible, generate as close to {solutions_to_generate} distinct, incorrect step-by-step solutions that demonstrate this specific error as possible.

        **Output Format:**
        Respond with a single JSON object matching the `TeacherResponse` schema.
        """

    # Acquire a permit from the rate limiter before making the API call
    self.rate_limiter.acquire()

    try:
      response = self.client.models.generate_content(
        model="gemini-2.5-flash-lite-preview-06-17",  # Using flash for speed and cost
        contents=prompt,
        config={
          "system_instruction": system_instruction,
          "response_mime_type": "application/json",
          "response_schema": TeacherResponse.model_json_schema(),
          "temperature": 0.7,  # Add some creativity to the generated errors
        },
      )
      assert response is not None, "API response is None"
      assert response.text is not None, "API response text is empty"
      teacher_response = TeacherResponse.model_validate_json(response.text)
      if teacher_response.is_valid_error:
        return teacher_response
      else:
        return None

    except (ValidationError, json.JSONDecodeError) as e:
      print(
        f"Error: Failed to parse or validate teacher response for concept {failure_concept.id}. Details: {e}"
      )
      return None
    except Exception as e:
      print(
        f"Error: An unexpected API error occurred for concept {failure_concept.id}. Details: {e}"
      )
      return None
