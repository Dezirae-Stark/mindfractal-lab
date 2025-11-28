"""
Core QWAMOS Engine and Base Classes

Implements quantum-inspired agent orchestration with superposition,
entanglement, and coherence concepts applied to distributed AI coordination.
"""

import asyncio
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np


class TaskState(Enum):
    """Quantum-inspired task states"""

    SUPERPOSITION = "superposition"  # Task exists in multiple potential states
    ENTANGLED = "entangled"  # Task is coupled with other tasks
    COLLAPSED = "collapsed"  # Task has been observed/executed
    COHERENT = "coherent"  # Task maintains phase relationship
    DECOHERENT = "decoherent"  # Task has lost quantum properties


class AgentState(Enum):
    """Quantum agent states"""

    GROUND = "ground"  # Lowest energy, ready state
    EXCITED = "excited"  # Active processing state
    SUPERPOSED = "superposed"  # Multiple simultaneous states
    ENTANGLED = "entangled"  # Correlated with other agents
    MEASURED = "measured"  # State has been observed


@dataclass
class QuantumTask:
    """Task with quantum properties"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    state: TaskState = TaskState.SUPERPOSITION
    priority: float = 0.5
    entangled_with: Set[str] = field(default_factory=set)
    state_vector: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0]))
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def collapse(self) -> "QuantumTask":
        """Collapse superposition to definite state"""
        self.state = TaskState.COLLAPSED
        # Normalize state vector
        self.state_vector = self.state_vector / np.linalg.norm(self.state_vector)
        return self

    def entangle_with(self, other_task_id: str):
        """Create entanglement with another task"""
        self.entangled_with.add(other_task_id)
        self.state = TaskState.ENTANGLED


@dataclass
class EntangledMessage:
    """Message with quantum correlations"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    receiver_id: str = ""
    content: Dict[str, Any] = field(default_factory=dict)
    entanglement_strength: float = 1.0
    phase: complex = 1.0 + 0j
    timestamp: datetime = field(default_factory=datetime.now)

    def measure(self) -> Dict[str, Any]:
        """Measure (read) the message, collapsing any superposition"""
        # Apply phase to content interpretation
        measured_content = self.content.copy()
        measured_content["_measurement_phase"] = self.phase
        measured_content["_entanglement_strength"] = self.entanglement_strength
        return measured_content


class QuantumAgent(ABC):
    """Base class for quantum-inspired agents"""

    def __init__(self, agent_id: str, name: str):
        self.id = agent_id
        self.name = name
        self.state = AgentState.GROUND
        self.state_vector = np.array([1.0, 0.0])  # |0⟩ state
        self.entangled_agents: Set[str] = set()
        self.coherence = 1.0
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.message_queue: asyncio.Queue = asyncio.Queue()

    @abstractmethod
    async def process_task(self, task: QuantumTask) -> Dict[str, Any]:
        """Process a quantum task"""
        pass

    @abstractmethod
    async def handle_message(self, message: EntangledMessage) -> Optional[EntangledMessage]:
        """Handle an entangled message"""
        pass

    async def excite(self):
        """Transition to excited state"""
        self.state = AgentState.EXCITED
        self.state_vector = np.array([0.0, 1.0])  # |1⟩ state

    async def superpose(self, amplitude1: complex, amplitude2: complex):
        """Enter superposition state"""
        self.state = AgentState.SUPERPOSED
        self.state_vector = np.array([amplitude1, amplitude2])
        self.state_vector = self.state_vector / np.linalg.norm(self.state_vector)

    def entangle_with(self, agent_id: str):
        """Create entanglement with another agent"""
        self.entangled_agents.add(agent_id)
        self.state = AgentState.ENTANGLED

    def measure(self) -> Tuple[AgentState, np.ndarray]:
        """Measure agent state, collapsing superposition"""
        if self.state == AgentState.SUPERPOSED:
            # Probabilistic collapse based on state vector
            prob_ground = abs(self.state_vector[0]) ** 2
            if np.random.random() < prob_ground:
                self.state = AgentState.GROUND
                self.state_vector = np.array([1.0, 0.0])
            else:
                self.state = AgentState.EXCITED
                self.state_vector = np.array([0.0, 1.0])

        self.state = AgentState.MEASURED
        return self.state, self.state_vector.copy()


class QWAMOSEngine:
    """
    Quantum Web Agent Multi-Operator System Engine

    Orchestrates multiple quantum agents with entanglement,
    superposition, and coherence properties.
    """

    def __init__(self, coherence_threshold: float = 0.7):
        self.agents: Dict[str, QuantumAgent] = {}
        self.tasks: Dict[str, QuantumTask] = {}
        self.entanglement_matrix: Dict[Tuple[str, str], complex] = {}
        self.coherence_threshold = coherence_threshold
        self.global_phase = 1.0 + 0j
        self.running = False

    def register_agent(self, agent: QuantumAgent):
        """Register a quantum agent"""
        self.agents[agent.id] = agent

    def create_entanglement(self, agent1_id: str, agent2_id: str, strength: complex = 1.0):
        """Create quantum entanglement between agents"""
        if agent1_id in self.agents and agent2_id in self.agents:
            self.agents[agent1_id].entangle_with(agent2_id)
            self.agents[agent2_id].entangle_with(agent1_id)
            self.entanglement_matrix[(agent1_id, agent2_id)] = strength
            self.entanglement_matrix[(agent2_id, agent1_id)] = strength.conjugate()

    async def submit_task(self, task: QuantumTask) -> str:
        """Submit a task to the quantum system"""
        self.tasks[task.id] = task

        # Distribute task based on superposition principle
        await self._distribute_task_superposition(task)

        return task.id

    async def _distribute_task_superposition(self, task: QuantumTask):
        """Distribute task across multiple agents in superposition"""
        # Calculate agent suitability amplitudes
        amplitudes = {}
        for agent_id, agent in self.agents.items():
            # Complex amplitude based on agent state and task priority
            amplitude = complex(
                task.priority * agent.coherence, (1 - task.priority) * (1 - agent.coherence)
            )
            amplitudes[agent_id] = amplitude

        # Normalize amplitudes
        total = sum(abs(a) ** 2 for a in amplitudes.values())
        if total > 0:
            for agent_id in amplitudes:
                amplitudes[agent_id] /= np.sqrt(total)

        # Create superposition of task across suitable agents
        for agent_id, amplitude in amplitudes.items():
            if abs(amplitude) > 0.1:  # Threshold for participation
                task_copy = QuantumTask(
                    id=f"{task.id}:{agent_id}",
                    description=task.description,
                    state=TaskState.SUPERPOSITION,
                    priority=task.priority,
                    state_vector=np.array([amplitude, 1 - amplitude]),
                )
                await self.agents[agent_id].task_queue.put(task_copy)

    async def broadcast_entangled_message(self, sender_id: str, content: Dict[str, Any]):
        """Broadcast message with quantum entanglement effects"""
        if sender_id not in self.agents:
            return

        sender = self.agents[sender_id]

        # Send to all entangled agents
        for receiver_id in sender.entangled_agents:
            if receiver_id in self.agents:
                # Calculate entanglement strength
                entanglement = self.entanglement_matrix.get((sender_id, receiver_id), 1.0 + 0j)

                message = EntangledMessage(
                    sender_id=sender_id,
                    receiver_id=receiver_id,
                    content=content,
                    entanglement_strength=abs(entanglement),
                    phase=entanglement / abs(entanglement) if abs(entanglement) > 0 else 1.0,
                )

                await self.agents[receiver_id].message_queue.put(message)

    async def measure_system_coherence(self) -> float:
        """Measure overall system coherence"""
        if not self.agents:
            return 0.0

        coherences = [agent.coherence for agent in self.agents.values()]

        # Consider entanglement in coherence calculation
        entanglement_factor = len(self.entanglement_matrix) / (len(self.agents) ** 2)

        return np.mean(coherences) * (1 + entanglement_factor)

    async def run(self):
        """Run the QWAMOS engine"""
        self.running = True

        # Start agent processing loops
        agent_tasks = []
        for agent in self.agents.values():
            agent_tasks.append(asyncio.create_task(self._agent_loop(agent)))

        # Monitor coherence
        monitor_task = asyncio.create_task(self._monitor_coherence())

        try:
            await asyncio.gather(*agent_tasks, monitor_task)
        except asyncio.CancelledError:
            self.running = False

    async def _agent_loop(self, agent: QuantumAgent):
        """Main processing loop for an agent"""
        while self.running:
            try:
                # Process tasks
                try:
                    task = await asyncio.wait_for(agent.task_queue.get(), timeout=0.1)
                    await agent.excite()
                    result = await agent.process_task(task)

                    # Handle task completion
                    if task.id in self.tasks:
                        self.tasks[task.id].collapse()

                except asyncio.TimeoutError:
                    pass

                # Process messages
                try:
                    message = await asyncio.wait_for(agent.message_queue.get(), timeout=0.1)
                    response = await agent.handle_message(message)

                    if response and response.receiver_id in self.agents:
                        await self.agents[response.receiver_id].message_queue.put(response)

                except asyncio.TimeoutError:
                    pass

                # Update coherence based on activity
                agent.coherence *= 0.99  # Gradual decoherence

                # Return to ground state if idle
                if agent.state == AgentState.EXCITED:
                    agent.state = AgentState.GROUND
                    agent.state_vector = np.array([1.0, 0.0])

            except Exception as e:
                print(f"Error in agent {agent.name}: {e}")

            await asyncio.sleep(0.01)  # Small delay to prevent CPU spinning

    async def _monitor_coherence(self):
        """Monitor and maintain system coherence"""
        while self.running:
            coherence = await self.measure_system_coherence()

            if coherence < self.coherence_threshold:
                # Re-establish coherence through phase alignment
                self.global_phase *= np.exp(1j * np.pi / 100)  # Small phase rotation

                # Boost agent coherence
                for agent in self.agents.values():
                    agent.coherence = min(1.0, agent.coherence * 1.01)

            await asyncio.sleep(1.0)

    async def shutdown(self):
        """Gracefully shutdown the engine"""
        self.running = False
        await asyncio.sleep(0.5)  # Allow loops to complete
