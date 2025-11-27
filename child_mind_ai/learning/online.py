"""
Child Mind AI — Online Learning
MindFractal Lab

Online learning and adaptation from experiences.
Updates internal parameters based on reward signals.
"""

import numpy as np
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import json
from pathlib import Path

from ..config import ChildMindAIConfig
from ..memory.episodic import EpisodicMemory, Episode
from ..memory.semantic import SemanticMemory


@dataclass
class LearningMetrics:
    """Metrics for tracking learning progress."""
    total_episodes: int = 0
    avg_reward: float = 0.0
    avg_coherence: float = 0.0
    reward_trend: float = 0.0  # Positive = improving
    coherence_trend: float = 0.0
    concepts_learned: int = 0
    patterns_learned: int = 0


class OnlineLearner:
    """
    Online learning system for Child Mind AI.

    Learns from each interaction to:
    - Update internal dynamics parameters
    - Form new concepts
    - Extract behavioral patterns
    - Improve response generation
    """

    def __init__(self, config: Optional[ChildMindAIConfig] = None):
        """Initialize online learner."""
        self.config = config or ChildMindAIConfig()

        # Memory systems
        self.episodic = EpisodicMemory(config)
        self.semantic = SemanticMemory(config)

        # Learning parameters
        self.learning_rate = config.learning_rate if config else 0.001
        self.batch_size = config.batch_size if config else 32
        self.update_frequency = config.update_frequency if config else 10

        # Learned parameter adjustments
        self.param_adjustments = self._load_adjustments()

        # Tracking
        self.episode_count = 0
        self.recent_rewards = []
        self.recent_coherences = []

    def _load_adjustments(self) -> Dict[str, float]:
        """Load learned parameter adjustments."""
        adjustments_file = self.config.memory_dir / "learned_adjustments.json"

        if adjustments_file.exists():
            try:
                with open(adjustments_file, 'r') as f:
                    return json.load(f)
            except:
                pass

        # Default adjustments (no change)
        return {
            "coherence_target_adj": 0.0,
            "memory_decay_adj": 0.0,
            "exploration_scale": 1.0,
            "stability_weight": 1.0,
        }

    def _save_adjustments(self):
        """Save learned parameter adjustments."""
        adjustments_file = self.config.memory_dir / "learned_adjustments.json"

        try:
            with open(adjustments_file, 'w') as f:
                json.dump(self.param_adjustments, f, indent=2)
        except Exception as e:
            print(f"[OnlineLearner] Warning: Could not save adjustments: {e}")

    def record_experience(
        self,
        state,  # ConsciousnessState
        input_text: str,
        input_encoding: np.ndarray,
        response_text: str,
        prev_coherence: float
    ) -> float:
        """
        Record an experience and compute reward.

        Returns:
            Computed reward signal
        """
        # Compute reward components
        coherence = float(state.c[0]) if len(state.c) > 0 else 0.5
        stability = float(state.c[3]) if len(state.c) > 3 else 0.5
        novelty = float(state.c[4]) if len(state.c) > 4 else 0.3

        # Reward function (matching Child Mind v1)
        target_coherence = self.config.coherence_target + self.param_adjustments["coherence_target_adj"]

        r_coh = np.exp(-0.5 * ((coherence - target_coherence) / 0.15) ** 2)
        r_nov = min(1.0, novelty * 2)  # Novelty capped at 1
        r_stab = stability

        # Weighted combination
        reward = 1.0 * r_coh + 0.5 * r_nov + 0.3 * r_stab

        # Store episode
        self.episodic.store(
            state=state,
            input_text=input_text,
            input_encoding=input_encoding,
            response_text=response_text,
            reward=reward,
            prev_coherence=prev_coherence
        )

        # Track metrics
        self.episode_count += 1
        self.recent_rewards.append(reward)
        self.recent_coherences.append(coherence)

        # Keep recent history bounded
        if len(self.recent_rewards) > 100:
            self.recent_rewards = self.recent_rewards[-100:]
            self.recent_coherences = self.recent_coherences[-100:]

        # Periodic learning updates
        if self.episode_count % self.update_frequency == 0:
            self._perform_learning_update()

        return reward

    def _perform_learning_update(self):
        """Perform periodic learning update."""
        # Sample batch from episodic memory
        batch = self.episodic.sample_batch(self.batch_size, strategy="reward")

        if len(batch) < 5:
            return  # Not enough data

        # Analyze batch for patterns
        self._update_param_adjustments(batch)

        # Form concepts from high-reward episodes
        high_reward = [ep for ep in batch if ep.reward > np.mean([e.reward for e in batch])]
        if len(high_reward) >= 3:
            self._form_concept_from_episodes(high_reward)

        # Extract patterns
        if len(batch) >= 10:
            self.semantic.learn_pattern(batch)

    def _update_param_adjustments(self, batch: List[Episode]):
        """Update learned parameter adjustments based on batch analysis."""
        rewards = np.array([ep.reward for ep in batch])
        coherences = np.array([ep.coherence for ep in batch])
        changes = np.array([ep.coherence_change for ep in batch])

        # Correlate reward with coherence
        if len(rewards) > 5:
            # If high coherence correlates with high reward, shift target up
            correlation = np.corrcoef(coherences, rewards)[0, 1]
            if not np.isnan(correlation):
                self.param_adjustments["coherence_target_adj"] += (
                    self.learning_rate * correlation * 0.1
                )
                # Clamp adjustment
                self.param_adjustments["coherence_target_adj"] = np.clip(
                    self.param_adjustments["coherence_target_adj"], -0.2, 0.2
                )

            # Analyze stability vs reward
            stabilities = np.array([ep.stability for ep in batch])
            stab_correlation = np.corrcoef(stabilities, rewards)[0, 1]
            if not np.isnan(stab_correlation):
                self.param_adjustments["stability_weight"] += (
                    self.learning_rate * stab_correlation * 0.5
                )
                self.param_adjustments["stability_weight"] = np.clip(
                    self.param_adjustments["stability_weight"], 0.5, 2.0
                )

        self._save_adjustments()

    def _form_concept_from_episodes(self, episodes: List[Episode]):
        """Form a semantic concept from a cluster of episodes."""
        # Get common tags
        tag_counts = {}
        for ep in episodes:
            for tag in ep.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        # Most common tag becomes concept name
        if tag_counts:
            concept_name = max(tag_counts, key=tag_counts.get)
        else:
            concept_name = f"concept_{len(self.semantic.concepts)}"

        # Use input encodings as exemplars
        exemplars = [np.array(ep.input_encoding) for ep in episodes]

        self.semantic.learn_concept(
            name=concept_name,
            description=f"Learned from {len(episodes)} high-reward interactions",
            exemplars=exemplars
        )

    def get_context_from_memory(
        self,
        input_encoding: np.ndarray,
        n_episodes: int = 3,
        n_concepts: int = 2
    ) -> Dict[str, Any]:
        """
        Retrieve relevant context from memory for response generation.

        Args:
            input_encoding: Current input encoding
            n_episodes: Number of similar episodes to retrieve
            n_concepts: Number of related concepts to retrieve

        Returns:
            Context dictionary with relevant memories
        """
        context = {
            "similar_episodes": [],
            "related_concepts": [],
            "applicable_patterns": [],
            "param_adjustments": self.param_adjustments.copy()
        }

        # Retrieve similar episodes
        similar = self.episodic.retrieve_similar(input_encoding, n_episodes)
        context["similar_episodes"] = [
            {
                "input": ep.input_text[:100],
                "response": ep.response_text[:100],
                "reward": ep.reward,
                "coherence": ep.coherence
            }
            for ep in similar
        ]

        # Find related concepts
        concept_matches = self.semantic.find_similar_concept(input_encoding, threshold=0.3)
        context["related_concepts"] = [
            {
                "name": concept.name,
                "confidence": concept.confidence,
                "similarity": sim
            }
            for concept, sim in concept_matches[:n_concepts]
        ]

        return context

    def get_metrics(self) -> LearningMetrics:
        """Get current learning metrics."""
        metrics = LearningMetrics()

        metrics.total_episodes = self.episode_count

        if self.recent_rewards:
            metrics.avg_reward = float(np.mean(self.recent_rewards))
            if len(self.recent_rewards) >= 20:
                first_half = np.mean(self.recent_rewards[:len(self.recent_rewards)//2])
                second_half = np.mean(self.recent_rewards[len(self.recent_rewards)//2:])
                metrics.reward_trend = second_half - first_half

        if self.recent_coherences:
            metrics.avg_coherence = float(np.mean(self.recent_coherences))
            if len(self.recent_coherences) >= 20:
                first_half = np.mean(self.recent_coherences[:len(self.recent_coherences)//2])
                second_half = np.mean(self.recent_coherences[len(self.recent_coherences)//2:])
                metrics.coherence_trend = second_half - first_half

        metrics.concepts_learned = len(self.semantic.concepts)
        metrics.patterns_learned = len(self.semantic.patterns)

        return metrics

    def describe_learning_state(self) -> str:
        """Generate natural language description of learning state."""
        metrics = self.get_metrics()

        parts = []

        if metrics.total_episodes == 0:
            return "I haven't learned from any interactions yet."

        parts.append(f"I've learned from {metrics.total_episodes} interactions.")

        if metrics.reward_trend > 0.05:
            parts.append("My performance has been improving.")
        elif metrics.reward_trend < -0.05:
            parts.append("I've been struggling recently.")
        else:
            parts.append("My performance has been stable.")

        if metrics.concepts_learned > 0:
            parts.append(f"I've formed {metrics.concepts_learned} concepts.")

        if metrics.patterns_learned > 0:
            parts.append(f"I've identified {metrics.patterns_learned} behavioral patterns.")

        return " ".join(parts)
