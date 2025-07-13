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

  def get_prerequisites(
    self, node_id: str, depth: int = 1
  ) -> List[ConceptNode]:
    """Retrieves the prerequisites for a given node at a specific depth.

    Args:
        node_id: The ID of the concept node.
        depth: The depth of prerequisites to retrieve. depth=1 returns
               direct prerequisites, depth=2 returns prerequisites of
               prerequisites, and so on.

    Returns:
        A list of unique ConceptNode objects at the specified prerequisite depth.
    """
    assert depth >= 1, "Depth must be at least 1"

    node = self.get_node(node_id)
    if not node:
      return []

    # Base case for recursion: depth is 1, return direct prerequisites
    if depth == 1:
      return [
        p_node
        for p in node.prerequisites
        if (p_node := self.get_node(p.concept_id))
      ]

    # Recursive step: get prerequisites from the next level down
    deeper_prereqs = []
    for direct_prereq in self.get_prerequisites(node_id, depth=1):
      deeper_prereqs.extend(self.get_prerequisites(direct_prereq.id, depth - 1))

    # Return a list of unique nodes by their ID
    return list({p.id: p for p in deeper_prereqs}.values())

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
