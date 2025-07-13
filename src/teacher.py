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


class PrerequisiteAnalysis(BaseModel):
  """A container for a single prerequisite's analysis."""

  concept_id: str
  response: TeacherResponse


class SingleProblemAnalysis(BaseModel):
  """Represents the analysis for one problem against all its prerequisites."""

  problem_str: str
  prerequisite_analyses: List[PrerequisiteAnalysis]


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

  def analyze_single_problem(
    self,
    problem_and_solution: ProblemSolutionPair,
    failure_concepts: List[ConceptNode],
    solutions_to_generate: int,
  ) -> SingleProblemAnalysis | None:
    """
    Analyzes a single problem against a batch of prerequisite concepts.

    Args:
        problem_and_solution: The problem-solution pair to analyze.
        failure_concepts: A list of prerequisite concepts to check for errors.
        solutions_to_generate: The number of incorrect solutions
                                   to generate for each plausible error.

    Returns:
        A SingleProblemAnalysis object containing the full analysis, or None if
        the API response is invalid.
    """
    system_instruction = (
      "You are an expert in math pedagogy. Your task is to perform a "
      "comprehensive analysis of a math problem against a batch of potential "
      "prerequisite errors. For EACH prerequisite, determine if making that "
      "error is a plausible reason for failing the problem. If it is, "
      "generate incorrect solutions. The output must be a single, nested "
      "JSON object."
    )

    problem_str = f'Problem: "{problem_and_solution.problem}"\nSolution: "{problem_and_solution.solution}"'

    concepts_str = ""
    for concept in failure_concepts:
      concepts_str += f'\n- Concept ID: "{concept.id}"\n  - Name: "{concept.name}"\n  - Description: "{concept.description}"'

    prompt = f"""
        Analyze the following problem against the batch of potential prerequisite errors.

        **Problem to Analyze:**
        {problem_str}

        **Batch of Potential Prerequisite Causes for Error:**
        {concepts_str}

        **Your Tasks:**
        For EACH prerequisite concept in the batch:
          1.  **Validation:** Is the prerequisite concept actively used in the correct solution for this specific problem? Is it plausible that a student could get this problem wrong because of a mistake in this specific prerequisite? Set `is_valid_error` to true or false and provide your `reasoning`.
          2.  **Generation:** If and only if the error is plausible (`is_valid_error: true`), generate exactly {solutions_to_generate} distinct, incorrect step-by-step solutions for this problem that demonstrate the specific prerequisite error.

        **Output Format:**
        Respond with a single JSON object matching the `SingleProblemAnalysis` schema.
        """

    # Acquire a permit from the rate limiter before making the API call
    self.rate_limiter.acquire()

    response = self.client.models.generate_content(
      model="gemini-2.5-flash",
      contents=prompt,
      config={
        "system_instruction": system_instruction,
        "response_mime_type": "application/json",
        "response_schema": SingleProblemAnalysis.model_json_schema(),
      },
    )
    assert response is not None, "API response is None"
    assert response.text is not None, "API response text is None"

    try:
      parsed_response = json.loads(response.text)
      return SingleProblemAnalysis.model_validate(parsed_response)
    except (ValidationError, json.JSONDecodeError) as e:
      print(f"Failed to parse or validate teacher response: {e}")
      return None
