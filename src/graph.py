"""Data models for the knowledge graph."""

import json
from pathlib import Path
from typing import Dict, List, Union
from collections import deque

from pydantic import BaseModel, Field


class PrerequisiteEdge(BaseModel):
  """Represents a prerequisite relationship between two concepts."""

  concept_id: str
  weight: float


class ProblemSolutionPair(BaseModel):
  """A container for a canonical problem and its correct, step-by-step solution."""

  problem: str
  solution: str


class ConceptNode(BaseModel):
  """Represents a single concept in the knowledge graph."""

  id: str
  name: str
  description: str
  problems_and_solutions: List[ProblemSolutionPair]
  prerequisites: List[PrerequisiteEdge] = Field(default_factory=list)


class KnowledgeGraph(BaseModel):
  """Represents the entire knowledge graph, composed of multiple concept nodes."""

  nodes: Dict[str, ConceptNode]

  def get_node(self, node_id: str) -> ConceptNode | None:
    """Retrieves a concept node by its ID.

    Args:
        node_id: The ID of the concept node to retrieve.

    Returns:
        The ConceptNode object if found, otherwise None.
    """
    return self.nodes.get(node_id)

  def get_prerequisites(self, node_id: str) -> List[ConceptNode]:
    """Retrieves the direct prerequisites for a given node.

    Args:
        node_id: The ID of the concept node.

    Returns:
        A list of ConceptNode objects that are direct prerequisites.
    """
    node = self.get_node(node_id)
    if not node:
      return []

    prerequisites = []
    for p in node.prerequisites:
      prerequisite_node = self.get_node(p.concept_id)
      if prerequisite_node is not None:
        prerequisites.append(prerequisite_node)

    return prerequisites

  @classmethod
  def load_from_json(cls, file_path: Union[str, Path]) -> "KnowledgeGraph":
    """Loads the knowledge graph from a JSON file.

    Args:
        file_path: The path to the JSON file.

    Returns:
        An instance of the KnowledgeGraph.
    """
    with open(file_path, "r") as f:
      data = json.load(f)

    nodes = {
      node_data["id"]: ConceptNode.model_validate(node_data)
      for node_data in data
    }
    return cls(nodes=nodes)

  def get_all_descendants(self, node_id: str) -> List[ConceptNode]:
    """
    Performs a traversal to get all unique prerequisite descendants for a node.

    Args:
        node_id: The ID of the node to start the traversal from.

    Returns:
        A flat list of all unique descendant ConceptNode objects.
    """
    descendants = []
    queue = deque([node_id])
    visited = {node_id}

    while queue:
      current_id = queue.popleft()
      current_node = self.get_node(current_id)
      if not current_node:
        continue

      for prereq_edge in current_node.prerequisites:
        prereq_id = prereq_edge.concept_id
        if prereq_id not in visited:
          visited.add(prereq_id)
          prereq_node = self.get_node(prereq_id)
          if prereq_node:
            descendants.append(prereq_node)
            queue.append(prereq_id)

    return descendants
