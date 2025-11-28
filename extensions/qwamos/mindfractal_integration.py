"""
MindFractal Integration Module for QWAMOS

Integrates the quantum multi-agent orchestration system with
the existing MindFractal Lab codebase for consciousness modeling.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from mindfractal import FractalDynamicsModel

from .agents import (
    Q0_MetaArchitect,
    Q1_MathematicalFormalist,
    Q2_ComputationalEngineer,
    Q3_DocumentationWeaver,
    Q4_VisualizationArtist,
    Q5_SystemsIntegrator,
    Q6_ConsciousnessModeler,
    Q7_FractalAnalyst,
)
from .core import QuantumTask, QWAMOSEngine, TaskState
from .protocols import (
    CoherenceProtocol,
    EntanglementConsensus,
    QuantumCommunicationProtocol,
    SuperpositionTaskDistribution,
)


class MindFractalQWAMOS:
    """
    Main integration class connecting QWAMOS to MindFractal Lab
    """

    def __init__(self):
        self.engine = QWAMOSEngine(coherence_threshold=0.7)
        self.agents_initialized = False
        self.protocols_initialized = False

        # Integration mappings
        self.task_mappings = {
            "design": ["Q0", "Q5"],  # Architecture and integration
            "mathematical": ["Q1", "Q7"],  # Math and fractal analysis
            "implement": ["Q2", "Q5"],  # Engineering and integration
            "document": ["Q3", "Q0"],  # Documentation and architecture
            "visualize": ["Q4", "Q6", "Q7"],  # Visualization, consciousness, fractals
            "analyze": ["Q6", "Q7", "Q1"],  # Analysis tasks
            "integrate": ["Q5", "Q0", "Q2"],  # Integration tasks
        }

    async def initialize(self):
        """Initialize QWAMOS agents and protocols"""
        # Create and register agents
        agents = [
            Q0_MetaArchitect(),
            Q1_MathematicalFormalist(),
            Q2_ComputationalEngineer(),
            Q3_DocumentationWeaver(),
            Q4_VisualizationArtist(),
            Q5_SystemsIntegrator(),
            Q6_ConsciousnessModeler(),
            Q7_FractalAnalyst(),
        ]

        for agent in agents:
            self.engine.register_agent(agent)

        # Create entanglements based on natural relationships
        entanglements = [
            ("Q0", "Q5", 0.9 + 0.1j),  # Architect <-> Integrator
            ("Q1", "Q7", 0.8 + 0.2j),  # Mathematician <-> Fractal Analyst
            ("Q6", "Q7", 0.85 + 0.15j),  # Consciousness <-> Fractal
            ("Q2", "Q5", 0.8 + 0.2j),  # Engineer <-> Integrator
            ("Q4", "Q6", 0.7 + 0.3j),  # Visualization <-> Consciousness
            ("Q0", "Q3", 0.7 + 0.3j),  # Architect <-> Documentation
        ]

        for agent1, agent2, strength in entanglements:
            self.engine.create_entanglement(agent1, agent2, strength)

        # Initialize protocols
        self.comm_protocol = QuantumCommunicationProtocol()
        self.task_distributor = SuperpositionTaskDistribution()
        self.consensus = EntanglementConsensus()
        self.coherence = CoherenceProtocol()

        # Establish communication channels
        for agent1_id in self.engine.agents:
            for agent2_id in self.engine.agents:
                if agent1_id < agent2_id:  # Avoid duplicates
                    self.comm_protocol.establish_channel(agent1_id, agent2_id)

        self.agents_initialized = True
        self.protocols_initialized = True

    async def create_consciousness_model(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new consciousness model using QWAMOS orchestration
        """
        if not self.agents_initialized:
            await self.initialize()

        # Create quantum tasks for the modeling process
        tasks = [
            QuantumTask(
                description="Design consciousness model architecture",
                priority=0.9,
                metadata={"parameters": parameters},
            ),
            QuantumTask(
                description="Formalize mathematical framework for consciousness dynamics",
                priority=0.9,
                metadata={"type": "nonlinear_dynamics"},
            ),
            QuantumTask(
                description="Implement fractal consciousness algorithms",
                priority=0.8,
                metadata={"optimization": "quantum_inspired"},
            ),
            QuantumTask(
                description="Analyze fractal properties of consciousness",
                priority=0.8,
                metadata={"dimensions": ["box_counting", "correlation", "quantum"]},
            ),
        ]

        # Create entanglements between related tasks
        tasks[0].entangle_with(tasks[1].id)
        tasks[1].entangle_with(tasks[2].id)
        tasks[2].entangle_with(tasks[3].id)

        # Submit tasks to engine
        task_ids = []
        for task in tasks:
            task_id = await self.engine.submit_task(task)
            task_ids.append(task_id)

        # Run engine asynchronously
        engine_task = asyncio.create_task(self._run_engine_cycle())

        # Wait for tasks to complete
        results = await self._collect_results(task_ids, timeout=30.0)

        # Cancel engine task
        engine_task.cancel()

        # Integrate results
        integrated_result = await self._integrate_consciousness_results(results)

        return integrated_result

    async def analyze_fractal_dynamics(self, model: FractalDynamicsModel) -> Dict[str, Any]:
        """
        Analyze fractal dynamics of a given model using QWAMOS
        """
        if not self.agents_initialized:
            await self.initialize()

        # Create analysis tasks
        tasks = [
            QuantumTask(
                description="Analyze mathematical properties of fractal dynamics",
                priority=0.85,
                metadata={"model_params": model.get_parameters()},
            ),
            QuantumTask(
                description="Visualize fractal attractor dynamics",
                priority=0.8,
                metadata={"visualization_type": "phase_space"},
            ),
            QuantumTask(
                description="Model consciousness emergence from fractal dynamics",
                priority=0.9,
                metadata={"consciousness_metric": "integrated_information"},
            ),
        ]

        # Submit and process
        task_ids = []
        for task in tasks:
            task_id = await self.engine.submit_task(task)
            task_ids.append(task_id)

        # Run analysis
        engine_task = asyncio.create_task(self._run_engine_cycle())
        results = await self._collect_results(task_ids, timeout=20.0)
        engine_task.cancel()

        # Combine analysis results
        analysis = {
            "mathematical_analysis": results.get(task_ids[0], {}),
            "visualization": results.get(task_ids[1], {}),
            "consciousness_emergence": results.get(task_ids[2], {}),
            "system_coherence": await self.engine.measure_system_coherence(),
        }

        return analysis

    async def orchestrate_research_task(self, task_description: str) -> Dict[str, Any]:
        """
        Orchestrate a general research task using QWAMOS
        """
        if not self.agents_initialized:
            await self.initialize()

        # Analyze task to determine which agents to involve
        primary_agents = self._determine_primary_agents(task_description)

        # Create quantum task
        task = QuantumTask(
            description=task_description, priority=0.85, metadata={"primary_agents": primary_agents}
        )

        # Submit task
        task_id = await self.engine.submit_task(task)

        # Process with quantum parallelism
        engine_task = asyncio.create_task(self._run_engine_cycle())

        # Wait for consensus among agents
        consensus_reached = False
        max_attempts = 10

        for attempt in range(max_attempts):
            await asyncio.sleep(2.0)  # Allow processing time

            # Check consensus
            consensus_reached, consensus_info = await self.consensus.achieve_consensus(
                self.engine.agents, task_description
            )

            if consensus_reached:
                break

        engine_task.cancel()

        # Collect final result
        if consensus_reached:
            result = {
                "status": "completed",
                "consensus": consensus_info,
                "task_id": task_id,
                "agents_involved": primary_agents,
                "quantum_advantage_utilized": True,
            }
        else:
            result = {
                "status": "partial",
                "reason": "Consensus not reached",
                "partial_results": consensus_info,
                "task_id": task_id,
            }

        return result

    async def _run_engine_cycle(self):
        """Run engine processing cycle"""
        try:
            await self.engine.run()
        except asyncio.CancelledError:
            await self.engine.shutdown()

    async def _collect_results(self, task_ids: List[str], timeout: float = 30.0) -> Dict[str, Any]:
        """Collect results from completed tasks"""
        results = {}
        start_time = asyncio.get_event_loop().time()

        while len(results) < len(task_ids):
            # Check timeout
            if asyncio.get_event_loop().time() - start_time > timeout:
                break

            # Check task states
            for task_id in task_ids:
                if task_id in self.engine.tasks:
                    task = self.engine.tasks[task_id]
                    if task.state == TaskState.COLLAPSED:
                        # Task completed, extract result
                        results[task_id] = task.metadata.get("result", {})

            await asyncio.sleep(0.5)

        return results

    async def _integrate_consciousness_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate results from consciousness modeling tasks"""
        integrated = {
            "model_type": "quantum_fractal_consciousness",
            "architecture": None,
            "mathematics": None,
            "implementation": None,
            "analysis": None,
        }

        # Extract components from results
        for task_id, result in results.items():
            if "design" in str(result):
                integrated["architecture"] = result
            elif "mathematical" in str(result):
                integrated["mathematics"] = result
            elif "implement" in str(result):
                integrated["implementation"] = result
            elif "analysis" in str(result):
                integrated["analysis"] = result

        # Add quantum properties
        integrated["quantum_properties"] = {
            "superposition_utilized": True,
            "entanglement_pattern": "hierarchical",
            "coherence_maintained": await self.coherence.monitor_coherence(self.engine.agents),
            "measurement_basis": "consciousness_observable",
        }

        return integrated

    def _determine_primary_agents(self, task_description: str) -> List[str]:
        """Determine which agents should primarily handle a task"""
        task_lower = task_description.lower()

        # Find matching task types
        primary_agents = []

        for task_type, agent_ids in self.task_mappings.items():
            if task_type in task_lower:
                primary_agents.extend(agent_ids)

        # Remove duplicates while preserving order
        seen = set()
        primary_agents = [x for x in primary_agents if not (x in seen or seen.add(x))]

        # Default to all agents if no specific match
        if not primary_agents:
            primary_agents = list(self.engine.agents.keys())

        return primary_agents

    async def get_system_status(self) -> Dict[str, Any]:
        """Get current QWAMOS system status"""
        if not self.agents_initialized:
            return {"status": "not_initialized"}

        agent_states = {}
        for agent_id, agent in self.engine.agents.items():
            agent_states[agent_id] = {
                "state": agent.state.value,
                "coherence": agent.coherence,
                "entangled_with": list(agent.entangled_agents),
            }

        return {
            "status": "operational",
            "agents": agent_states,
            "active_tasks": len(self.engine.tasks),
            "system_coherence": await self.engine.measure_system_coherence(),
            "entanglement_density": len(self.engine.entanglement_matrix)
            / len(self.engine.agents) ** 2,
        }


# Convenience functions for integration
async def create_qwamos_consciousness_model(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Create consciousness model using QWAMOS orchestration"""
    qwamos = MindFractalQWAMOS()
    return await qwamos.create_consciousness_model(parameters)


async def analyze_with_qwamos(model: FractalDynamicsModel) -> Dict[str, Any]:
    """Analyze fractal dynamics model with QWAMOS"""
    qwamos = MindFractalQWAMOS()
    return await qwamos.analyze_fractal_dynamics(model)


async def orchestrate_task(description: str) -> Dict[str, Any]:
    """Orchestrate a research task with QWAMOS"""
    qwamos = MindFractalQWAMOS()
    return await qwamos.orchestrate_research_task(description)
