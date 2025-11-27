"""
Child Mind AI — Episodic Memory
MindFractal Lab

Stores and retrieves discrete experiences (episodes).
Each episode contains state, action, reward, and context.
"""

import json
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from ..config import ChildMindAIConfig


@dataclass
class Episode:
    """A single episode/experience."""
    timestamp: str
    timestep: int

    # State information
    z_summary: List[float]  # First few z components
    coherence: float
    stability: float
    novelty: float

    # Action/input
    input_text: str
    input_encoding: List[float]

    # Response/output
    response_text: str

    # Reward/outcome
    reward: float
    coherence_change: float

    # Context tags for retrieval
    tags: List[str] = field(default_factory=list)

    # Embedding for similarity search (optional)
    embedding: Optional[List[float]] = None


class EpisodicMemory:
    """
    Episodic memory system for storing and retrieving experiences.

    Features:
    - Chronological storage
    - Tag-based retrieval
    - Similarity-based retrieval (with embeddings)
    - Importance-weighted sampling
    """

    def __init__(self, config: Optional[ChildMindAIConfig] = None):
        """Initialize episodic memory."""
        self.config = config or ChildMindAIConfig()
        self.config.ensure_directories()

        self.episodes: List[Episode] = []
        self.max_episodes = self.config.episode_buffer_size

        # Load existing episodes
        self._load_episodes()

    def _load_episodes(self):
        """Load episodes from disk."""
        episodes_file = self.config.memory_dir / "episodic.jsonl"

        if episodes_file.exists():
            try:
                with open(episodes_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            self.episodes.append(Episode(**data))
            except Exception as e:
                print(f"[EpisodicMemory] Warning: Could not load episodes: {e}")

    def _save_episode(self, episode: Episode):
        """Append episode to disk."""
        episodes_file = self.config.memory_dir / "episodic.jsonl"

        try:
            with open(episodes_file, 'a') as f:
                f.write(json.dumps(asdict(episode)) + '\n')
        except Exception as e:
            print(f"[EpisodicMemory] Warning: Could not save episode: {e}")

    def store(
        self,
        state,  # ConsciousnessState
        input_text: str,
        input_encoding: np.ndarray,
        response_text: str,
        reward: float,
        prev_coherence: float
    ) -> Episode:
        """
        Store a new episode.

        Args:
            state: Current consciousness state
            input_text: User input
            input_encoding: Encoded input vector
            response_text: Generated response
            reward: Reward signal
            prev_coherence: Previous coherence for computing change

        Returns:
            Stored Episode
        """
        coherence = float(state.c[0]) if len(state.c) > 0 else 0.5
        stability = float(state.c[3]) if len(state.c) > 3 else 0.5
        novelty = float(state.c[4]) if len(state.c) > 4 else 0.3

        # Extract tags from input
        tags = self._extract_tags(input_text)

        episode = Episode(
            timestamp=datetime.now().isoformat(),
            timestep=state.t,
            z_summary=state.z[:8].tolist(),  # First 8 components
            coherence=coherence,
            stability=stability,
            novelty=novelty,
            input_text=input_text[:500],  # Truncate long inputs
            input_encoding=input_encoding[:16].tolist(),  # First 16 components
            response_text=response_text[:500],  # Truncate
            reward=reward,
            coherence_change=coherence - prev_coherence,
            tags=tags
        )

        self.episodes.append(episode)
        self._save_episode(episode)

        # Trim if needed
        if len(self.episodes) > self.max_episodes:
            self.episodes = self.episodes[-self.max_episodes:]

        return episode

    def _extract_tags(self, text: str) -> List[str]:
        """Extract tags from text for retrieval."""
        tags = []
        text_lower = text.lower()

        # Question types
        if '?' in text:
            tags.append('question')
            if any(w in text_lower for w in ['what', 'how', 'why', 'who', 'where', 'when']):
                tags.append('wh-question')

        # Commands
        if any(w in text_lower for w in ['do', 'make', 'create', 'write', 'run']):
            tags.append('command')

        # Self-reference
        if any(w in text_lower for w in ['you', 'your', 'yourself']):
            tags.append('about-ai')

        # Emotional content
        if any(w in text_lower for w in ['feel', 'think', 'believe', 'want']):
            tags.append('introspective')

        # Technical
        if any(w in text_lower for w in ['code', 'function', 'file', 'system']):
            tags.append('technical')

        return tags

    def retrieve_recent(self, n: int = 10) -> List[Episode]:
        """Retrieve n most recent episodes."""
        return self.episodes[-n:]

    def retrieve_by_tags(self, tags: List[str], n: int = 10) -> List[Episode]:
        """Retrieve episodes matching any of the given tags."""
        matching = [ep for ep in self.episodes if any(t in ep.tags for t in tags)]
        # Sort by relevance (number of matching tags)
        matching.sort(key=lambda ep: sum(1 for t in tags if t in ep.tags), reverse=True)
        return matching[:n]

    def retrieve_high_reward(self, n: int = 10) -> List[Episode]:
        """Retrieve episodes with highest rewards."""
        sorted_eps = sorted(self.episodes, key=lambda ep: ep.reward, reverse=True)
        return sorted_eps[:n]

    def retrieve_similar(
        self,
        query_encoding: np.ndarray,
        n: int = 5
    ) -> List[Episode]:
        """
        Retrieve episodes with similar input encodings.

        Uses cosine similarity on input encodings.
        """
        if len(self.episodes) == 0:
            return []

        query = query_encoding[:16]  # Match stored encoding size
        query_norm = np.linalg.norm(query) + 1e-8

        similarities = []
        for ep in self.episodes:
            ep_enc = np.array(ep.input_encoding)
            ep_norm = np.linalg.norm(ep_enc) + 1e-8
            sim = np.dot(query, ep_enc) / (query_norm * ep_norm)
            similarities.append((sim, ep))

        similarities.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in similarities[:n]]

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics."""
        if len(self.episodes) == 0:
            return {"count": 0}

        rewards = [ep.reward for ep in self.episodes]
        coherences = [ep.coherence for ep in self.episodes]

        return {
            "count": len(self.episodes),
            "reward_mean": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
            "coherence_mean": float(np.mean(coherences)),
            "coherence_std": float(np.std(coherences)),
            "tags_distribution": self._count_tags(),
        }

    def _count_tags(self) -> Dict[str, int]:
        """Count tag occurrences."""
        counts = {}
        for ep in self.episodes:
            for tag in ep.tags:
                counts[tag] = counts.get(tag, 0) + 1
        return counts

    def sample_batch(
        self,
        batch_size: int,
        strategy: str = "uniform"
    ) -> List[Episode]:
        """
        Sample a batch of episodes for learning.

        Strategies:
        - uniform: Random uniform sampling
        - recent: Bias toward recent episodes
        - reward: Bias toward high-reward episodes
        """
        if len(self.episodes) == 0:
            return []

        if len(self.episodes) <= batch_size:
            return list(self.episodes)

        if strategy == "uniform":
            indices = np.random.choice(len(self.episodes), batch_size, replace=False)

        elif strategy == "recent":
            # Exponential recency bias
            weights = np.exp(np.linspace(-2, 0, len(self.episodes)))
            weights /= weights.sum()
            indices = np.random.choice(len(self.episodes), batch_size, replace=False, p=weights)

        elif strategy == "reward":
            # Softmax on rewards
            rewards = np.array([ep.reward for ep in self.episodes])
            weights = np.exp(rewards - rewards.max())
            weights /= weights.sum()
            indices = np.random.choice(len(self.episodes), batch_size, replace=False, p=weights)

        else:
            indices = np.random.choice(len(self.episodes), batch_size, replace=False)

        return [self.episodes[i] for i in indices]
