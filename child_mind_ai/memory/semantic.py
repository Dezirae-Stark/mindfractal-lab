"""
Child Mind AI — Semantic Memory
MindFractal Lab

Long-term storage of distilled knowledge and patterns.
Learns generalizations from episodic experiences.
"""

import json
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from ..config import ChildMindAIConfig


@dataclass
class SemanticConcept:
    """A learned concept or pattern."""
    name: str
    description: str

    # Prototype vector (average of associated experiences)
    prototype: List[float]

    # Variance for each dimension
    variance: List[float]

    # Association strength
    confidence: float

    # Related concepts
    associations: Dict[str, float] = field(default_factory=dict)

    # Usage statistics
    access_count: int = 0
    last_accessed: str = ""

    # Creation info
    created: str = ""
    updated: str = ""


@dataclass
class Pattern:
    """A learned behavioral pattern (state transition)."""
    name: str

    # Trigger conditions (when does this pattern apply?)
    trigger_coherence_range: tuple  # (min, max)
    trigger_stability_range: tuple
    trigger_tags: List[str]

    # Expected outcome
    expected_coherence_change: float
    expected_reward: float

    # Confidence
    observation_count: int
    confidence: float


class SemanticMemory:
    """
    Semantic memory for storing long-term knowledge.

    Features:
    - Concept formation from episodes
    - Pattern extraction
    - Associative retrieval
    - Knowledge distillation
    """

    def __init__(self, config: Optional[ChildMindAIConfig] = None):
        """Initialize semantic memory."""
        self.config = config or ChildMindAIConfig()
        self.config.ensure_directories()

        self.concepts: Dict[str, SemanticConcept] = {}
        self.patterns: List[Pattern] = []

        self._load()

    def _load(self):
        """Load semantic memory from disk."""
        concepts_file = self.config.memory_dir / "semantic_concepts.json"
        patterns_file = self.config.memory_dir / "semantic_patterns.json"

        if concepts_file.exists():
            try:
                with open(concepts_file, 'r') as f:
                    data = json.load(f)
                    for name, concept_data in data.items():
                        self.concepts[name] = SemanticConcept(**concept_data)
            except Exception as e:
                print(f"[SemanticMemory] Warning: Could not load concepts: {e}")

        if patterns_file.exists():
            try:
                with open(patterns_file, 'r') as f:
                    data = json.load(f)
                    self.patterns = [Pattern(**p) for p in data]
            except Exception as e:
                print(f"[SemanticMemory] Warning: Could not load patterns: {e}")

    def save(self):
        """Save semantic memory to disk."""
        concepts_file = self.config.memory_dir / "semantic_concepts.json"
        patterns_file = self.config.memory_dir / "semantic_patterns.json"

        try:
            with open(concepts_file, 'w') as f:
                json.dump({k: asdict(v) for k, v in self.concepts.items()}, f, indent=2)

            with open(patterns_file, 'w') as f:
                json.dump([asdict(p) for p in self.patterns], f, indent=2)
        except Exception as e:
            print(f"[SemanticMemory] Warning: Could not save: {e}")

    def learn_concept(
        self,
        name: str,
        description: str,
        exemplars: List[np.ndarray]
    ) -> SemanticConcept:
        """
        Learn or update a concept from exemplar vectors.

        Args:
            name: Concept name
            description: Human-readable description
            exemplars: List of feature vectors representing the concept

        Returns:
            Created/updated SemanticConcept
        """
        if len(exemplars) == 0:
            return None

        # Compute prototype (centroid)
        exemplars_array = np.array(exemplars)
        prototype = np.mean(exemplars_array, axis=0)
        variance = np.var(exemplars_array, axis=0)

        # Confidence based on number of exemplars and variance
        n = len(exemplars)
        avg_variance = np.mean(variance)
        confidence = min(1.0, n / 10) * max(0.1, 1 - avg_variance)

        now = datetime.now().isoformat()

        if name in self.concepts:
            # Update existing concept
            existing = self.concepts[name]

            # Merge prototypes (weighted by observation count)
            old_weight = existing.access_count / (existing.access_count + n)
            new_weight = n / (existing.access_count + n)

            merged_prototype = (
                old_weight * np.array(existing.prototype) +
                new_weight * prototype
            )

            existing.prototype = merged_prototype.tolist()
            existing.variance = variance.tolist()
            existing.confidence = (existing.confidence + confidence) / 2
            existing.access_count += n
            existing.updated = now

            concept = existing
        else:
            # Create new concept
            concept = SemanticConcept(
                name=name,
                description=description,
                prototype=prototype.tolist(),
                variance=variance.tolist(),
                confidence=confidence,
                created=now,
                updated=now
            )
            self.concepts[name] = concept

        self.save()
        return concept

    def learn_pattern(
        self,
        episodes: List['Episode']  # from episodic memory
    ) -> Optional[Pattern]:
        """
        Extract a behavioral pattern from episodes.

        Groups similar episodes and extracts transition patterns.
        """
        if len(episodes) < 3:
            return None

        # Compute statistics
        coherences = [ep.coherence for ep in episodes]
        stabilities = [ep.stability for ep in episodes]
        changes = [ep.coherence_change for ep in episodes]
        rewards = [ep.reward for ep in episodes]

        # Extract common tags
        all_tags = {}
        for ep in episodes:
            for tag in ep.tags:
                all_tags[tag] = all_tags.get(tag, 0) + 1

        common_tags = [t for t, c in all_tags.items() if c >= len(episodes) * 0.5]

        # Create pattern
        pattern = Pattern(
            name=f"pattern_{len(self.patterns)}",
            trigger_coherence_range=(min(coherences), max(coherences)),
            trigger_stability_range=(min(stabilities), max(stabilities)),
            trigger_tags=common_tags,
            expected_coherence_change=float(np.mean(changes)),
            expected_reward=float(np.mean(rewards)),
            observation_count=len(episodes),
            confidence=min(1.0, len(episodes) / 20)
        )

        self.patterns.append(pattern)
        self.save()
        return pattern

    def retrieve_concept(self, name: str) -> Optional[SemanticConcept]:
        """Retrieve a concept by name."""
        if name in self.concepts:
            concept = self.concepts[name]
            concept.access_count += 1
            concept.last_accessed = datetime.now().isoformat()
            return concept
        return None

    def find_similar_concept(
        self,
        query_vector: np.ndarray,
        threshold: float = 0.5
    ) -> List[tuple]:
        """
        Find concepts similar to a query vector.

        Returns list of (concept, similarity) tuples.
        """
        results = []

        for name, concept in self.concepts.items():
            prototype = np.array(concept.prototype)

            # Truncate to match sizes
            min_len = min(len(query_vector), len(prototype))
            q = query_vector[:min_len]
            p = prototype[:min_len]

            # Cosine similarity
            sim = np.dot(q, p) / (np.linalg.norm(q) + 1e-8) / (np.linalg.norm(p) + 1e-8)

            if sim >= threshold:
                results.append((concept, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def find_applicable_pattern(
        self,
        coherence: float,
        stability: float,
        tags: List[str]
    ) -> List[Pattern]:
        """Find patterns that might apply to current state."""
        applicable = []

        for pattern in self.patterns:
            # Check coherence range
            if not (pattern.trigger_coherence_range[0] <= coherence <= pattern.trigger_coherence_range[1]):
                continue

            # Check stability range
            if not (pattern.trigger_stability_range[0] <= stability <= pattern.trigger_stability_range[1]):
                continue

            # Check tag overlap
            if pattern.trigger_tags:
                overlap = len(set(tags) & set(pattern.trigger_tags))
                if overlap == 0:
                    continue

            applicable.append(pattern)

        # Sort by confidence
        applicable.sort(key=lambda p: p.confidence, reverse=True)
        return applicable

    def add_association(
        self,
        concept1: str,
        concept2: str,
        strength: float = 0.5
    ):
        """Add or strengthen association between concepts."""
        if concept1 in self.concepts and concept2 in self.concepts:
            self.concepts[concept1].associations[concept2] = min(
                1.0,
                self.concepts[concept1].associations.get(concept2, 0) + strength
            )
            self.concepts[concept2].associations[concept1] = min(
                1.0,
                self.concepts[concept2].associations.get(concept1, 0) + strength
            )
            self.save()

    def get_associated_concepts(
        self,
        name: str,
        threshold: float = 0.3
    ) -> List[tuple]:
        """Get concepts associated with the given concept."""
        if name not in self.concepts:
            return []

        concept = self.concepts[name]
        associated = [
            (self.concepts[n], s)
            for n, s in concept.associations.items()
            if n in self.concepts and s >= threshold
        ]
        associated.sort(key=lambda x: x[1], reverse=True)
        return associated

    def get_statistics(self) -> Dict[str, Any]:
        """Get semantic memory statistics."""
        return {
            "concept_count": len(self.concepts),
            "pattern_count": len(self.patterns),
            "avg_concept_confidence": np.mean([c.confidence for c in self.concepts.values()]) if self.concepts else 0,
            "avg_pattern_confidence": np.mean([p.confidence for p in self.patterns]) if self.patterns else 0,
        }
