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


class PortfolioResponse(BaseModel):
  """The top-level response model for a portfolio of problems."""

  portfolio_analysis: List[SingleProblemAnalysis]


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

  def analyze_problem_portfolio(
    self,
    problems_and_solutions: List[ProblemSolutionPair],
    failure_concepts: List[ConceptNode],
    max_solutions_to_generate: int,
  ) -> PortfolioResponse | None:
    """
    Analyzes a portfolio of problems against a batch of prerequisite concepts.

    Args:
        problems_and_solutions: A list of problem-solution pairs to analyze.
        failure_concepts: A list of prerequisite concepts to check for errors.
        max_solutions_to_generate: The maximum number of incorrect solutions
                                   to generate for each plausible error.

    Returns:
        A PortfolioResponse object containing the full analysis, or None if
        the API response is invalid.
    """
    system_instruction = (
      "You are an expert in math pedagogy. Your task is to perform a "
      "comprehensive analysis of a portfolio of math problems against a batch "
      "of potential prerequisite errors. For EACH problem, you must analyze it "
      "against EACH prerequisite. For every combination, determine if making "
      "the prerequisite error is a plausible reason for failing that specific "
      "problem. If it is, generate incorrect solutions. The output must be a "
      "single, nested JSON object."
    )

    problems_str = ""
    for i, ps in enumerate(problems_and_solutions):
      problems_str += (
        f'\n- Problem {i + 1}: "{ps.problem}"\n  - Solution: "{ps.solution}"'
      )

    concepts_str = ""
    for concept in failure_concepts:
      concepts_str += f'\n- Concept ID: "{concept.id}"\n  - Name: "{concept.name}"\n  - Description: "{concept.description}"'

    prompt = f"""
        Analyze the following portfolio of problems against the batch of potential prerequisite errors.

        **Problem Portfolio:**
        {problems_str}

        **Batch of Potential Prerequisite Causes for Error:**
        {concepts_str}

        **Your Tasks:**
        For EACH problem in the portfolio:
          For EACH prerequisite concept in the batch:
            1.  **Validation:** Is the prerequisite concept actively used in the correct solution for this specific problem? Is it plausible that a student could get this problem wrong because of a mistake in this specific prerequisite? Set `is_valid_error` to true or false and provide your `reasoning`.
            2.  **Generation:** If and only if the error is plausible (`is_valid_error: true`), generate up to {max_solutions_to_generate} distinct, incorrect step-by-step solutions for this problem that demonstrate the specific prerequisite error.

        **Output Format:**
        Respond with a single JSON object matching the `PortfolioResponse` schema.
        """

    # Acquire a permit from the rate limiter before making the API call
    self.rate_limiter.acquire()

    response = self.client.models.generate_content(
      model="gemini-2.5-flash",
      contents=prompt,
      config={
        "system_instruction": system_instruction,
        "response_mime_type": "application/json",
        "response_schema": PortfolioResponse.model_json_schema(),
      },
    )
    assert response is not None, "API response is None"
    assert response.text is not None, "API response text is None"

    try:
      parsed_response = json.loads(response.text)
      return PortfolioResponse.model_validate(parsed_response)
    except (ValidationError, json.JSONDecodeError) as e:
      print(f"Failed to parse or validate teacher response: {e}")
      return None
