import os
from google import genai
from pydantic import BaseModel, TypeAdapter
from typing import List
from graph import ConceptNode


class GeneratedErrorDetail(BaseModel):
  """Details of a generated synthetic solution."""

  step_number: int
  incorrect_solution: List[str]


class TeacherModel:
  """A model representing the 'Teacher' component for generating synthetic errors."""

  def __init__(self):
    """Initializes the TeacherModel and configures the Gemini API client."""
    if not os.getenv("GEMINI_API_KEY"):
      raise ValueError("GEMINI_API_KEY environment variable not set.")

    self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

  def generate_synthetic_errors(
    self,
    problem_example: str,
    correct_solution: str,
    failure_concept: ConceptNode,
  ) -> list[GeneratedErrorDetail]:
    """
    Generates a synthetic, step-by-step incorrect solution to a math problem.

    This method prompts the Gemini model to act as a student who understands
    the target concept but makes a specific error related to a prerequisite
    (the failure concept).

    Args:
        problem_example: An example problem related to the target concept.
        correct_solution: The correct, step-by-step solution to the problem.
        failure_concept_name: The name of the prerequisite concept the student is failing at.

    Returns:
        A Recipe object containing the validation and generated solutions.
    """

    system_instruction = (
      "You are an expert in math pedagogy. Your task is to analyze a math problem, its correct "
      "solution, and a potential prerequisite error. First, you must determine if making the "
      "specified prerequisite error is a plausible reason for failing the main problem. "
      "If it is, you will select the steps at which the error could occur, "
      "then, for each selected step, act as a flawed student and generate a solution that "
      "makes the specific prerequisite error. The output must consist of a list of "
      "erroneous solutions."
    )

    prompt = f"""
        Analyze the following scenario:

        **Problem:**
        {problem_example}

        **Correct Step-by-Step Solution:**
        {correct_solution}

        **Potential Prerequisite Cause for Error:**
        name: {failure_concept.name}
        description: {failure_concept.description}

        **Your Tasks:**
        1.  **Validation:** Is `{failure_concept.name}` a concept that is actively used in the correct solution? Is it plausible that a student could get the main problem wrong because of a mistake in this specific prerequisite?
        2.  **Generation:** If and only if the error is plausible, identify 1 to 3 steps where the error could occur, then generate distinct, incorrect step-by-step solutions. Each solution should demonstrate the specific error related to `{failure_concept.name}`.

        **Output Format:**
        Respond with a list of GeneratedErrorDetails, where each GeneratedErrorDetail contains the step number (step_number: int) where the error occurs and the corresponding incorrect solution (incorrect_solution: List[str]).
        Each incorrect solution should be a list of strings, where each string represents a step in the flawed student's reasoning.
        """

    response = self.client.models.generate_content(
      model="gemini-2.5-flash",
      contents=prompt,
      config={
        "system_instruction": system_instruction,
        "response_mime_type": "application/json",
        "response_schema": list[GeneratedErrorDetail],
      },
    )
    my_recipes: list[GeneratedErrorDetail] = TypeAdapter(
      list[GeneratedErrorDetail]
    ).validate_python(response.parsed)
    return my_recipes
