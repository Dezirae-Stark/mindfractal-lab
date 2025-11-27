"""
Quantum Utilities for QWAMOS

Provides quantum-inspired mathematical tools and utilities for
the multi-agent orchestration system.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import scipy.linalg


@dataclass
class StateVector:
    """
    Quantum state vector representation
    """
    amplitudes: np.ndarray
    basis_labels: Optional[List[str]] = None
    
    def __post_init__(self):
        # Ensure normalization
        self.normalize()
        
    def normalize(self):
        """Normalize the state vector"""
        norm = np.linalg.norm(self.amplitudes)
        if norm > 1e-10:
            self.amplitudes = self.amplitudes / norm
            
    def measure(self, basis: Optional[np.ndarray] = None) -> Tuple[int, float]:
        """
        Perform quantum measurement
        
        Returns: (outcome_index, probability)
        """
        if basis is not None:
            # Transform to measurement basis
            transformed = basis @ self.amplitudes
            probabilities = np.abs(transformed)**2
        else:
            # Computational basis measurement
            probabilities = np.abs(self.amplitudes)**2
            
        # Normalize probabilities
        probabilities = probabilities / np.sum(probabilities)
        
        # Perform measurement
        outcome = np.random.choice(len(probabilities), p=probabilities)
        
        return outcome, probabilities[outcome]
        
    def apply_gate(self, gate: np.ndarray) -> 'StateVector':
        """Apply quantum gate to state"""
        new_amplitudes = gate @ self.amplitudes
        return StateVector(new_amplitudes, self.basis_labels)
        
    def tensor_product(self, other: 'StateVector') -> 'StateVector':
        """Compute tensor product with another state"""
        new_amplitudes = np.kron(self.amplitudes, other.amplitudes)
        
        # Combine basis labels if available
        if self.basis_labels and other.basis_labels:
            new_labels = [
                f"{l1}⊗{l2}" 
                for l1 in self.basis_labels 
                for l2 in other.basis_labels
            ]
        else:
            new_labels = None
            
        return StateVector(new_amplitudes, new_labels)
        
    def partial_trace(self, keep_dims: List[int], total_dims: int) -> np.ndarray:
        """
        Compute partial trace to get reduced density matrix
        """
        # Convert to density matrix
        rho = np.outer(self.amplitudes, self.amplitudes.conj())
        
        # Reshape for partial trace
        dim = 2  # Assuming qubits
        shape = [dim] * (2 * total_dims)
        rho_reshaped = rho.reshape(shape)
        
        # Trace out unwanted dimensions
        trace_dims = []
        for i in range(total_dims):
            if i not in keep_dims:
                trace_dims.append(i)
                trace_dims.append(i + total_dims)
                
        # Perform einsum trace
        einsum_str = self._build_einsum_string(total_dims, trace_dims)
        reduced_rho = np.einsum(einsum_str, rho_reshaped)
        
        # Reshape back to matrix
        reduced_dim = dim ** len(keep_dims)
        return reduced_rho.reshape(reduced_dim, reduced_dim)
        
    def _build_einsum_string(self, total_dims: int, trace_dims: List[int]) -> str:
        """Build einsum string for partial trace"""
        # This is a simplified version
        indices = list(range(2 * total_dims))
        
        # Set traced indices to same value
        for i in range(0, len(trace_dims), 2):
            indices[trace_dims[i+1]] = indices[trace_dims[i]]
            
        # Convert to letters
        letters = [chr(ord('a') + i) for i in indices]
        input_str = ''.join(letters)
        
        # Output string excludes traced indices
        output_letters = [letters[i] for i in range(2 * total_dims) 
                         if i not in trace_dims]
        output_str = ''.join(output_letters)
        
        return f"{input_str}->{output_str}"
        
    def entropy(self) -> float:
        """Calculate von Neumann entropy"""
        # For pure state, entropy is 0
        return 0.0
        
    def fidelity(self, other: 'StateVector') -> float:
        """Calculate fidelity with another state"""
        return abs(np.dot(self.amplitudes.conj(), other.amplitudes))**2


class ObservationOperator:
    """
    Quantum observation operator for measurements
    """
    
    def __init__(self, matrix: np.ndarray, name: str = "Observable"):
        self.matrix = matrix
        self.name = name
        
        # Ensure Hermitian
        if not np.allclose(matrix, matrix.conj().T):
            raise ValueError("Observable must be Hermitian")
            
        # Cache eigendecomposition
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(matrix)
        
    def expectation_value(self, state: StateVector) -> float:
        """Calculate expectation value in given state"""
        return np.real(
            np.dot(state.amplitudes.conj(), 
                  self.matrix @ state.amplitudes)
        )
        
    def measure(self, state: StateVector) -> Tuple[float, StateVector]:
        """
        Perform measurement on state
        
        Returns: (measurement_outcome, collapsed_state)
        """
        # Transform to eigenbasis
        transformed = self.eigenvectors.conj().T @ state.amplitudes
        probabilities = np.abs(transformed)**2
        
        # Perform measurement
        outcome_idx = np.random.choice(len(probabilities), p=probabilities)
        outcome_value = self.eigenvalues[outcome_idx]
        
        # Collapse state
        collapsed_amplitudes = self.eigenvectors[:, outcome_idx]
        collapsed_state = StateVector(collapsed_amplitudes.copy())
        
        return outcome_value, collapsed_state
        
    def uncertainty(self, state: StateVector) -> float:
        """Calculate uncertainty (standard deviation) in given state"""
        exp_val = self.expectation_value(state)
        exp_val_sq = np.real(
            np.dot(state.amplitudes.conj(),
                  self.matrix @ self.matrix @ state.amplitudes)
        )
        
        return np.sqrt(max(0, exp_val_sq - exp_val**2))


class EntanglementMatrix:
    """
    Represents and analyzes quantum entanglement between subsystems
    """
    
    def __init__(self, n_agents: int):
        self.n_agents = n_agents
        self.matrix = np.zeros((n_agents, n_agents), dtype=complex)
        
    def set_entanglement(self, agent1: int, agent2: int, strength: complex):
        """Set entanglement strength between two agents"""
        self.matrix[agent1, agent2] = strength
        self.matrix[agent2, agent1] = strength.conj()
        
    def get_entanglement(self, agent1: int, agent2: int) -> complex:
        """Get entanglement strength between two agents"""
        return self.matrix[agent1, agent2]
        
    def cluster_coefficient(self, agent: int) -> float:
        """
        Calculate clustering coefficient for an agent
        (measures local entanglement density)
        """
        neighbors = np.where(np.abs(self.matrix[agent]) > 1e-10)[0]
        n_neighbors = len(neighbors)
        
        if n_neighbors < 2:
            return 0.0
            
        # Count triangles
        triangles = 0
        for i in range(n_neighbors):
            for j in range(i+1, n_neighbors):
                if abs(self.matrix[neighbors[i], neighbors[j]]) > 1e-10:
                    triangles += 1
                    
        possible_triangles = n_neighbors * (n_neighbors - 1) / 2
        
        return triangles / possible_triangles if possible_triangles > 0 else 0.0
        
    def global_entanglement(self) -> float:
        """Calculate global entanglement measure"""
        # Use matrix norm as measure
        return np.linalg.norm(self.matrix, 'fro') / self.n_agents
        
    def find_clusters(self, threshold: float = 0.5) -> List[List[int]]:
        """Find entanglement clusters using threshold"""
        adjacency = np.abs(self.matrix) > threshold
        
        clusters = []
        visited = set()
        
        for agent in range(self.n_agents):
            if agent not in visited:
                cluster = self._dfs_cluster(agent, adjacency, visited)
                if len(cluster) > 1:  # Only keep non-trivial clusters
                    clusters.append(cluster)
                    
        return clusters
        
    def _dfs_cluster(self, start: int, adjacency: np.ndarray, 
                     visited: set) -> List[int]:
        """Depth-first search to find connected cluster"""
        cluster = []
        stack = [start]
        
        while stack:
            agent = stack.pop()
            if agent in visited:
                continue
                
            visited.add(agent)
            cluster.append(agent)
            
            # Add connected agents
            for neighbor in range(self.n_agents):
                if adjacency[agent, neighbor] and neighbor not in visited:
                    stack.append(neighbor)
                    
        return cluster
        
    def to_graph_laplacian(self) -> np.ndarray:
        """Convert to graph Laplacian for spectral analysis"""
        degrees = np.sum(np.abs(self.matrix), axis=1)
        laplacian = np.diag(degrees) - self.matrix
        return laplacian
        
    def algebraic_connectivity(self) -> float:
        """
        Calculate algebraic connectivity (Fiedler value)
        Measures how well-connected the entanglement network is
        """
        laplacian = self.to_graph_laplacian()
        eigenvalues = np.linalg.eigvalsh(laplacian)
        
        # Second smallest eigenvalue
        eigenvalues.sort()
        return eigenvalues[1] if len(eigenvalues) > 1 else 0.0


class CoherenceMetric:
    """
    Measures and tracks quantum coherence in the system
    """
    
    def __init__(self):
        self.history: List[float] = []
        self.timestamps: List[float] = []
        
    def measure_state_coherence(self, state: StateVector) -> float:
        """
        Measure coherence of a quantum state
        Uses l1-norm of coherence
        """
        # Convert to density matrix
        rho = np.outer(state.amplitudes, state.amplitudes.conj())
        
        # l1-norm of off-diagonal elements
        coherence = 0.0
        n = len(state.amplitudes)
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    coherence += abs(rho[i, j])
                    
        return coherence
        
    def measure_relative_entropy_coherence(self, state: StateVector) -> float:
        """
        Measure coherence using relative entropy
        More sophisticated measure
        """
        # Convert to density matrix
        rho = np.outer(state.amplitudes, state.amplitudes.conj())
        
        # Diagonal part (incoherent state)
        rho_diag = np.diag(np.diag(rho))
        
        # Relative entropy S(rho || rho_diag)
        # For numerical stability
        epsilon = 1e-10
        
        # Eigenvalues of rho and rho_diag
        eig_rho = np.linalg.eigvalsh(rho + epsilon * np.eye(len(rho)))
        eig_diag = np.diag(rho_diag) + epsilon
        
        # Calculate relative entropy
        rel_entropy = 0.0
        for p in eig_rho:
            if p > epsilon:
                # Sum over q values
                for q in eig_diag:
                    if q > epsilon:
                        rel_entropy += p * np.log(p / q)
                        
        return max(0, rel_entropy)
        
    def track_coherence(self, coherence: float, timestamp: float):
        """Track coherence over time"""
        self.history.append(coherence)
        self.timestamps.append(timestamp)
        
        # Keep only recent history (last 1000 points)
        if len(self.history) > 1000:
            self.history.pop(0)
            self.timestamps.pop(0)
            
    def decoherence_rate(self) -> float:
        """Estimate decoherence rate from history"""
        if len(self.history) < 2:
            return 0.0
            
        # Fit exponential decay
        # log(C) = log(C0) - gamma * t
        
        log_coherences = [np.log(c + 1e-10) for c in self.history]
        times = np.array(self.timestamps) - self.timestamps[0]
        
        # Linear fit
        A = np.vstack([times, np.ones(len(times))]).T
        gamma, log_c0 = np.linalg.lstsq(A, log_coherences, rcond=None)[0]
        
        return -gamma if gamma < 0 else 0.0
        
    def predict_coherence(self, future_time: float) -> float:
        """Predict future coherence based on decay rate"""
        if len(self.history) < 2:
            return self.history[-1] if self.history else 1.0
            
        rate = self.decoherence_rate()
        current_coherence = self.history[-1]
        time_diff = future_time - self.timestamps[-1]
        
        return current_coherence * np.exp(-rate * time_diff)


# Quantum Gates
class QuantumGates:
    """Common quantum gates for state manipulation"""
    
    # Pauli gates
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    # Hadamard gate
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    
    # Phase gates
    S = np.array([[1, 0], [0, 1j]], dtype=complex)
    T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
    
    @staticmethod
    def rotation_x(theta: float) -> np.ndarray:
        """X-axis rotation gate"""
        return np.array([
            [np.cos(theta/2), -1j * np.sin(theta/2)],
            [-1j * np.sin(theta/2), np.cos(theta/2)]
        ], dtype=complex)
        
    @staticmethod
    def rotation_y(theta: float) -> np.ndarray:
        """Y-axis rotation gate"""
        return np.array([
            [np.cos(theta/2), -np.sin(theta/2)],
            [np.sin(theta/2), np.cos(theta/2)]
        ], dtype=complex)
        
    @staticmethod
    def rotation_z(theta: float) -> np.ndarray:
        """Z-axis rotation gate"""
        return np.array([
            [np.exp(-1j * theta/2), 0],
            [0, np.exp(1j * theta/2)]
        ], dtype=complex)
        
    @staticmethod
    def cnot() -> np.ndarray:
        """Controlled-NOT gate"""
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=complex)
        
    @staticmethod
    def cz() -> np.ndarray:
        """Controlled-Z gate"""
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1]
        ], dtype=complex)