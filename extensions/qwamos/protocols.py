"""
Quantum Communication and Coordination Protocols for QWAMOS

Implements quantum-inspired protocols for agent communication,
task distribution, and consensus mechanisms.
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Set, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import asyncio
from enum import Enum

from .core import QuantumAgent, QuantumTask, EntangledMessage, TaskState, AgentState


class ProtocolType(Enum):
    """Types of quantum protocols"""
    BELL_STATE = "bell_state"          # Maximum entanglement
    GHZ_STATE = "ghz_state"            # Multi-party entanglement  
    W_STATE = "w_state"                # Robust entanglement
    CLUSTER_STATE = "cluster_state"    # Computation-ready entanglement


@dataclass
class QuantumChannel:
    """Quantum communication channel between agents"""
    agent1_id: str
    agent2_id: str
    capacity: float = 1.0              # Channel capacity (0-1)
    noise_level: float = 0.0           # Quantum noise (0-1)
    entanglement_fidelity: float = 1.0 # Entanglement quality (0-1)
    
    def transmit(self, message: EntangledMessage) -> EntangledMessage:
        """Transmit message through quantum channel with noise"""
        # Apply quantum noise
        if self.noise_level > 0:
            # Degrade entanglement strength
            message.entanglement_strength *= (1 - self.noise_level)
            
            # Add phase noise
            phase_noise = np.exp(1j * np.random.normal(0, self.noise_level))
            message.phase *= phase_noise
            
        # Apply capacity limitations
        message.entanglement_strength *= self.capacity
        
        return message


class QuantumCommunicationProtocol:
    """
    Quantum-inspired communication protocol for agent coordination
    """
    
    def __init__(self, protocol_type: ProtocolType = ProtocolType.BELL_STATE):
        self.protocol_type = protocol_type
        self.channels: Dict[Tuple[str, str], QuantumChannel] = {}
        self.message_buffer: Dict[str, List[EntangledMessage]] = {}
        
    def establish_channel(self, agent1_id: str, agent2_id: str, 
                         capacity: float = 1.0, noise: float = 0.0) -> QuantumChannel:
        """Establish quantum communication channel between agents"""
        channel = QuantumChannel(
            agent1_id=agent1_id,
            agent2_id=agent2_id,
            capacity=capacity,
            noise_level=noise
        )
        
        # Bidirectional channel
        self.channels[(agent1_id, agent2_id)] = channel
        self.channels[(agent2_id, agent1_id)] = channel
        
        return channel
        
    async def send_message(self, message: EntangledMessage) -> bool:
        """Send entangled message through quantum channel"""
        channel_key = (message.sender_id, message.receiver_id)
        
        if channel_key not in self.channels:
            # No direct channel, try quantum teleportation
            return await self._quantum_teleport(message)
            
        channel = self.channels[channel_key]
        transmitted = channel.transmit(message)
        
        # Buffer message for receiver
        if message.receiver_id not in self.message_buffer:
            self.message_buffer[message.receiver_id] = []
        self.message_buffer[message.receiver_id].append(transmitted)
        
        return True
        
    async def receive_messages(self, agent_id: str) -> List[EntangledMessage]:
        """Receive all messages for an agent"""
        messages = self.message_buffer.get(agent_id, [])
        self.message_buffer[agent_id] = []  # Clear buffer
        return messages
        
    async def broadcast_entangled_state(self, sender_id: str, 
                                      state: np.ndarray,
                                      receiver_ids: List[str]):
        """Broadcast quantum state to multiple agents"""
        if self.protocol_type == ProtocolType.GHZ_STATE:
            # Create GHZ state for multi-party entanglement
            n_parties = len(receiver_ids) + 1
            ghz_state = np.zeros(2**n_parties, dtype=complex)
            ghz_state[0] = 1/np.sqrt(2)
            ghz_state[-1] = 1/np.sqrt(2)
            
            # Send correlated messages
            for i, receiver_id in enumerate(receiver_ids):
                message = EntangledMessage(
                    sender_id=sender_id,
                    receiver_id=receiver_id,
                    content={"shared_state": state.tolist(), "ghz_index": i},
                    entanglement_strength=1.0,
                    phase=ghz_state[i] if i < len(ghz_state) else 1.0
                )
                await self.send_message(message)
                
        elif self.protocol_type == ProtocolType.W_STATE:
            # Create W state for robust entanglement
            n_parties = len(receiver_ids) + 1
            
            for i, receiver_id in enumerate(receiver_ids):
                # W state has equal superposition of single excitations
                amplitude = 1/np.sqrt(n_parties)
                message = EntangledMessage(
                    sender_id=sender_id,
                    receiver_id=receiver_id,
                    content={"shared_state": state.tolist(), "w_index": i},
                    entanglement_strength=abs(amplitude),
                    phase=np.exp(2j * np.pi * i / n_parties)
                )
                await self.send_message(message)
                
    async def _quantum_teleport(self, message: EntangledMessage) -> bool:
        """Quantum teleportation protocol for indirect communication"""
        # Find intermediate agent with channels to both sender and receiver
        for (a1, a2), channel in self.channels.items():
            if a1 == message.sender_id and a2 in self.channels:
                # Check if intermediate agent has channel to receiver
                if (a2, message.receiver_id) in self.channels:
                    # Teleport through intermediate agent
                    intermediate_msg = EntangledMessage(
                        sender_id=message.sender_id,
                        receiver_id=a2,
                        content={"teleport": True, "final_dest": message.receiver_id,
                                **message.content},
                        entanglement_strength=message.entanglement_strength * 0.9,
                        phase=message.phase
                    )
                    
                    await self.send_message(intermediate_msg)
                    
                    # Forward from intermediate to final destination
                    final_msg = EntangledMessage(
                        sender_id=a2,
                        receiver_id=message.receiver_id,
                        content=message.content,
                        entanglement_strength=message.entanglement_strength * 0.8,
                        phase=message.phase * np.exp(1j * np.pi/4)  # Teleportation phase
                    )
                    
                    await self.send_message(final_msg)
                    return True
                    
        return False


class SuperpositionTaskDistribution:
    """
    Distribute tasks using quantum superposition principle
    """
    
    def __init__(self, coherence_threshold: float = 0.5):
        self.coherence_threshold = coherence_threshold
        self.task_amplitudes: Dict[str, Dict[str, complex]] = {}
        
    async def distribute_task(self, task: QuantumTask, 
                            agents: Dict[str, QuantumAgent]) -> Dict[str, float]:
        """
        Distribute task across agents in superposition
        
        Returns probability distribution of task assignment
        """
        amplitudes = {}
        
        for agent_id, agent in agents.items():
            # Calculate amplitude based on agent state and task priority
            if agent.state == AgentState.GROUND:
                base_amplitude = 0.9
            elif agent.state == AgentState.EXCITED:
                base_amplitude = 0.3  # Already busy
            else:
                base_amplitude = 0.6
                
            # Modulate by coherence and task priority
            amplitude = complex(
                base_amplitude * agent.coherence * task.priority,
                np.sqrt(1 - agent.coherence**2) * (1 - task.priority)
            )
            
            # Consider entanglement effects
            if task.entangled_with:
                # Boost amplitude if agent is entangled with related tasks
                entanglement_boost = len(
                    task.entangled_with.intersection(agent.entangled_agents)
                ) * 0.1
                amplitude *= (1 + entanglement_boost)
                
            amplitudes[agent_id] = amplitude
            
        # Normalize amplitudes
        total = sum(abs(a)**2 for a in amplitudes.values())
        if total > 0:
            probabilities = {
                agent_id: abs(amp)**2 / total 
                for agent_id, amp in amplitudes.items()
            }
        else:
            # Equal distribution if no preferences
            probabilities = {
                agent_id: 1/len(agents) for agent_id in agents
            }
            
        # Store amplitudes for later collapse
        self.task_amplitudes[task.id] = amplitudes
        
        return probabilities
        
    async def collapse_distribution(self, task_id: str, 
                                  selected_agent_id: str) -> float:
        """
        Collapse superposition to specific agent assignment
        
        Returns collapse fidelity (0-1)
        """
        if task_id not in self.task_amplitudes:
            return 0.0
            
        amplitudes = self.task_amplitudes[task_id]
        
        if selected_agent_id not in amplitudes:
            return 0.0
            
        # Calculate fidelity of collapse
        selected_amplitude = amplitudes[selected_agent_id]
        total = sum(abs(a)**2 for a in amplitudes.values())
        
        fidelity = abs(selected_amplitude)**2 / total if total > 0 else 0.0
        
        # Clean up
        del self.task_amplitudes[task_id]
        
        return fidelity
        
    def calculate_interference(self, task1_id: str, task2_id: str) -> complex:
        """Calculate quantum interference between two task distributions"""
        if task1_id not in self.task_amplitudes or task2_id not in self.task_amplitudes:
            return 0.0 + 0j
            
        amps1 = self.task_amplitudes[task1_id]
        amps2 = self.task_amplitudes[task2_id]
        
        # Calculate overlap (inner product)
        interference = 0.0 + 0j
        for agent_id in set(amps1.keys()).intersection(amps2.keys()):
            interference += amps1[agent_id].conjugate() * amps2[agent_id]
            
        return interference


class EntanglementConsensus:
    """
    Quantum entanglement-based consensus protocol
    """
    
    def __init__(self, entanglement_threshold: float = 0.7):
        self.entanglement_threshold = entanglement_threshold
        self.consensus_states: Dict[str, np.ndarray] = {}
        
    async def achieve_consensus(self, agents: Dict[str, QuantumAgent],
                               topic: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Achieve consensus through quantum entanglement
        
        Returns (consensus_reached, consensus_state)
        """
        n_agents = len(agents)
        if n_agents == 0:
            return False, {}
            
        # Initialize consensus state vector
        consensus_dim = 2**n_agents
        consensus_state = np.zeros(consensus_dim, dtype=complex)
        
        # Build entangled state from agent states
        agent_list = list(agents.values())
        
        for i in range(consensus_dim):
            amplitude = 1.0 + 0j
            
            for j, agent in enumerate(agent_list):
                # Extract bit for this agent
                agent_bit = (i >> j) & 1
                
                # Multiply by agent's state amplitude
                if agent_bit == 0:
                    amplitude *= agent.state_vector[0]
                else:
                    amplitude *= agent.state_vector[1]
                    
            consensus_state[i] = amplitude
            
        # Normalize
        norm = np.linalg.norm(consensus_state)
        if norm > 0:
            consensus_state /= norm
            
        # Calculate entanglement measure (concurrence)
        entanglement = self._calculate_concurrence(consensus_state, n_agents)
        
        # Store consensus state
        self.consensus_states[topic] = consensus_state
        
        # Determine if consensus reached
        consensus_reached = bool(entanglement >= self.entanglement_threshold)
        
        # Extract consensus decision
        if consensus_reached:
            # Measure consensus state
            probabilities = np.abs(consensus_state)**2
            max_idx = np.argmax(probabilities)
            
            # Decode decision from binary representation
            decision = {
                agent.id: bool((max_idx >> i) & 1)
                for i, agent in enumerate(agent_list)
            }
            
            consensus_info = {
                "decision": decision,
                "entanglement": float(entanglement),
                "confidence": float(probabilities[max_idx]),
                "state_vector": consensus_state.tolist()
            }
        else:
            consensus_info = {
                "decision": None,
                "entanglement": float(entanglement),
                "confidence": 0.0,
                "reason": "Insufficient entanglement"
            }
            
        return consensus_reached, consensus_info
        
    def _calculate_concurrence(self, state: np.ndarray, n_qubits: int) -> float:
        """Calculate concurrence as measure of entanglement"""
        # Simplified concurrence for multi-qubit state
        # Real implementation would use reduced density matrices
        
        # Check for product state (no entanglement)
        is_product = False
        for i in range(len(state)):
            if abs(state[i]) > 0.99:  # Nearly pure state
                is_product = True
                break
                
        if is_product:
            return 0.0
            
        # Estimate entanglement from state distribution
        probs = np.abs(state)**2
        probs = probs[probs > 1e-10]
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        max_entropy = np.log(2**n_qubits)
        
        return min(1.0, entropy / max_entropy)
        
    async def synchronize_phase(self, agents: Dict[str, QuantumAgent]):
        """Synchronize quantum phases across all agents"""
        if not agents:
            return
            
        # Calculate average phase
        avg_phase = np.mean([
            np.angle(agent.state_vector[1]) 
            for agent in agents.values()
            if abs(agent.state_vector[1]) > 1e-10
        ])
        
        # Apply phase correction to each agent
        phase_correction = np.exp(-1j * avg_phase)
        
        for agent in agents.values():
            agent.state_vector *= phase_correction


class CoherenceProtocol:
    """
    Protocol for maintaining quantum coherence in the system
    """
    
    def __init__(self, decoherence_rate: float = 0.01):
        self.decoherence_rate = decoherence_rate
        self.coherence_history: Dict[str, List[float]] = {}
        self.error_correction_codes: Dict[str, np.ndarray] = {}
        
    async def monitor_coherence(self, agents: Dict[str, QuantumAgent]) -> float:
        """Monitor system-wide coherence"""
        coherences = []
        
        for agent_id, agent in agents.items():
            coherences.append(agent.coherence)
            
            # Track history
            if agent_id not in self.coherence_history:
                self.coherence_history[agent_id] = []
            self.coherence_history[agent_id].append(agent.coherence)
            
            # Keep only recent history
            if len(self.coherence_history[agent_id]) > 100:
                self.coherence_history[agent_id].pop(0)
                
        return np.mean(coherences) if coherences else 0.0
        
    async def apply_error_correction(self, agent: QuantumAgent):
        """Apply quantum error correction to maintain coherence"""
        # Simplified error correction using amplitude damping channel
        
        # Calculate error probability
        error_prob = 1 - agent.coherence
        
        if error_prob > 0.5:
            # High error - reset to ground state
            agent.state_vector = np.array([1.0, 0.0])
            agent.coherence = 1.0
        else:
            # Apply amplitude damping correction
            e0 = np.array([[1, 0], [0, np.sqrt(1 - error_prob)]])
            e1 = np.array([[0, np.sqrt(error_prob)], [0, 0]])
            
            # Apply Kraus operators
            rho = np.outer(agent.state_vector, agent.state_vector.conj())
            rho_corrected = e0 @ rho @ e0.conj().T + e1 @ rho @ e1.conj().T
            
            # Extract corrected state (assuming pure state)
            eigenvals, eigenvecs = np.linalg.eigh(rho_corrected)
            max_idx = np.argmax(eigenvals)
            agent.state_vector = eigenvecs[:, max_idx]
            
            # Boost coherence
            agent.coherence = min(1.0, agent.coherence * 1.1)
            
    async def inject_fresh_coherence(self, agents: Dict[str, QuantumAgent]):
        """Inject fresh coherence into the system"""
        for agent in agents.values():
            if agent.coherence < 0.5:
                # Agent needs coherence boost
                agent.coherence = min(1.0, agent.coherence + 0.2)
                
                # Also refresh phase
                random_phase = np.exp(2j * np.pi * np.random.rand())
                agent.state_vector[1] *= random_phase
                
    def predict_decoherence_time(self, agent: QuantumAgent) -> float:
        """Predict time until agent loses coherence"""
        if agent.id not in self.coherence_history:
            return float('inf')
            
        history = self.coherence_history[agent.id]
        if len(history) < 2:
            return float('inf')
            
        # Fit exponential decay
        decay_rate = np.mean([
            history[i] - history[i-1] 
            for i in range(1, len(history))
        ])
        
        if decay_rate >= 0:
            return float('inf')
            
        # Time to reach coherence = 0.1
        time_to_decoherence = (0.1 - agent.coherence) / decay_rate
        
        return max(0, time_to_decoherence)