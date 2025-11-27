"""
Test Suite for QWAMOS Integration

Tests the quantum multi-agent orchestration system functionality
and integration with MindFractal Lab.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

sys.path.append(str(Path(__file__).parent.parent))

from extensions.qwamos import (
    CoherenceMetric,
    CoherenceProtocol,
    EntangledMessage,
    EntanglementConsensus,
    EntanglementMatrix,
    ObservationOperator,
    Q0_MetaArchitect,
    Q1_MathematicalFormalist,
    Q2_ComputationalEngineer,
    Q3_DocumentationWeaver,
    Q4_VisualizationArtist,
    Q5_SystemsIntegrator,
    Q6_ConsciousnessModeler,
    Q7_FractalAnalyst,
    QuantumCommunicationProtocol,
    QuantumTask,
    QWAMOSEngine,
    StateVector,
    SuperpositionTaskDistribution,
)
from extensions.qwamos.mindfractal_integration import MindFractalQWAMOS


class TestQuantumCore:
    """Test core quantum components"""

    def test_quantum_task_creation(self):
        """Test QuantumTask creation and properties"""
        task = QuantumTask(description="Test quantum task", priority=0.8)

        assert task.description == "Test quantum task"
        assert task.priority == 0.8
        assert task.state == task.state.SUPERPOSITION
        assert len(task.state_vector) == 2

    def test_quantum_task_collapse(self):
        """Test task collapse from superposition"""
        task = QuantumTask()
        task.state_vector = np.array([0.6, 0.8])

        collapsed = task.collapse()

        assert task.state == task.state.COLLAPSED
        assert np.abs(np.linalg.norm(task.state_vector) - 1.0) < 1e-10

    def test_entangled_message(self):
        """Test EntangledMessage creation and measurement"""
        message = EntangledMessage(
            sender_id="Q1",
            receiver_id="Q2",
            content={"data": "quantum"},
            entanglement_strength=0.9,
            phase=np.exp(1j * np.pi / 4),
        )

        measured = message.measure()

        assert "data" in measured
        assert measured["data"] == "quantum"
        assert "_measurement_phase" in measured
        assert "_entanglement_strength" in measured

    @pytest.mark.asyncio
    async def test_quantum_agent_state_transitions(self):
        """Test agent state transitions"""
        agent = Q0_MetaArchitect()

        # Initial state
        assert agent.state == agent.state.GROUND

        # Excite
        await agent.excite()
        assert agent.state == agent.state.EXCITED
        assert np.allclose(agent.state_vector, [0, 1])

        # Superpose
        await agent.superpose(1 / np.sqrt(2), 1 / np.sqrt(2))
        assert agent.state == agent.state.SUPERPOSED
        assert np.abs(np.linalg.norm(agent.state_vector) - 1.0) < 1e-10

    def test_agent_entanglement(self):
        """Test agent entanglement"""
        agent1 = Q1_MathematicalFormalist()
        agent2 = Q2_ComputationalEngineer()

        agent1.entangle_with(agent2.id)

        assert agent2.id in agent1.entangled_agents
        assert agent1.state == agent1.state.ENTANGLED


class TestQWAMOSEngine:
    """Test QWAMOS engine functionality"""

    @pytest.mark.asyncio
    async def test_engine_initialization(self):
        """Test engine initialization and agent registration"""
        engine = QWAMOSEngine()

        agent1 = Q0_MetaArchitect()
        agent2 = Q1_MathematicalFormalist()

        engine.register_agent(agent1)
        engine.register_agent(agent2)

        assert len(engine.agents) == 2
        assert "Q0" in engine.agents
        assert "Q1" in engine.agents

    @pytest.mark.asyncio
    async def test_task_submission(self):
        """Test task submission to engine"""
        engine = QWAMOSEngine()

        agent = Q2_ComputationalEngineer()
        engine.register_agent(agent)

        task = QuantumTask(description="Implement quantum algorithm")
        task_id = await engine.submit_task(task)

        assert task_id in engine.tasks
        assert engine.tasks[task_id].description == "Implement quantum algorithm"

    @pytest.mark.asyncio
    async def test_entanglement_creation(self):
        """Test creating entanglement between agents"""
        engine = QWAMOSEngine()

        agent1 = Q3_DocumentationWeaver()
        agent2 = Q4_VisualizationArtist()

        engine.register_agent(agent1)
        engine.register_agent(agent2)

        engine.create_entanglement(agent1.id, agent2.id, 0.8 + 0.2j)

        assert agent2.id in agent1.entangled_agents
        assert agent1.id in agent2.entangled_agents
        assert (agent1.id, agent2.id) in engine.entanglement_matrix

    @pytest.mark.asyncio
    async def test_system_coherence(self):
        """Test system coherence measurement"""
        engine = QWAMOSEngine()

        agents = [
            Q5_SystemsIntegrator(),
            Q6_ConsciousnessModeler(),
            Q7_FractalAnalyst(),
        ]

        for agent in agents:
            engine.register_agent(agent)

        coherence = await engine.measure_system_coherence()

        assert 0.0 <= coherence <= 1.5  # With entanglement factor


class TestQuantumProtocols:
    """Test quantum communication and coordination protocols"""

    def test_quantum_channel(self):
        """Test quantum communication channel"""
        protocol = QuantumCommunicationProtocol()

        channel = protocol.establish_channel("Q1", "Q2", capacity=0.9, noise=0.1)

        assert channel.agent1_id == "Q1"
        assert channel.agent2_id == "Q2"
        assert channel.capacity == 0.9
        assert channel.noise_level == 0.1

    @pytest.mark.asyncio
    async def test_message_transmission(self):
        """Test message transmission through quantum channel"""
        protocol = QuantumCommunicationProtocol()
        protocol.establish_channel("Q1", "Q2")

        message = EntangledMessage(
            sender_id="Q1",
            receiver_id="Q2",
            content={"test": "data"},
            entanglement_strength=1.0,
        )

        success = await protocol.send_message(message)
        assert success

        messages = await protocol.receive_messages("Q2")
        assert len(messages) == 1
        assert messages[0].content["test"] == "data"

    @pytest.mark.asyncio
    async def test_superposition_distribution(self):
        """Test task distribution in superposition"""
        distributor = SuperpositionTaskDistribution()

        agents = {
            "Q1": Mock(
                state=Mock(GROUND="ground"), coherence=0.9, entangled_agents=set()
            ),
            "Q2": Mock(
                state=Mock(EXCITED="excited"), coherence=0.8, entangled_agents=set()
            ),
            "Q3": Mock(
                state=Mock(GROUND="ground"), coherence=0.95, entangled_agents=set()
            ),
        }

        # Set agent states
        agents["Q1"].state = agents["Q1"].state.GROUND
        agents["Q2"].state = agents["Q2"].state.EXCITED
        agents["Q3"].state = agents["Q3"].state.GROUND

        task = QuantumTask(description="Test task", priority=0.7)

        probabilities = await distributor.distribute_task(task, agents)

        assert len(probabilities) == 3
        assert sum(probabilities.values()) == pytest.approx(1.0)
        assert all(0 <= p <= 1 for p in probabilities.values())

    @pytest.mark.asyncio
    async def test_entanglement_consensus(self):
        """Test consensus through entanglement"""
        consensus = EntanglementConsensus()

        agents = {
            "Q1": Mock(id="Q1", state_vector=np.array([1, 0])),
            "Q2": Mock(id="Q2", state_vector=np.array([0, 1])),
            "Q3": Mock(
                id="Q3", state_vector=np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])
            ),
        }

        reached, info = await consensus.achieve_consensus(agents, "test_topic")

        assert isinstance(reached, bool)
        assert "entanglement" in info
        assert "confidence" in info


class TestQuantumUtils:
    """Test quantum utility classes"""

    def test_state_vector(self):
        """Test StateVector operations"""
        state = StateVector(np.array([1, 1]))

        # Check normalization
        assert np.abs(np.linalg.norm(state.amplitudes) - 1.0) < 1e-10

        # Test measurement
        outcome, prob = state.measure()
        assert outcome in [0, 1]
        assert 0 <= prob <= 1

    def test_observation_operator(self):
        """Test ObservationOperator"""
        # Pauli Z operator
        z_op = ObservationOperator(np.array([[1, 0], [0, -1]]), "Z")

        state = StateVector(np.array([1, 0]))  # |0⟩ state

        # Expectation value
        exp_val = z_op.expectation_value(state)
        assert exp_val == pytest.approx(1.0)

        # Measurement
        outcome, collapsed = z_op.measure(state)
        assert outcome in [1, -1]
        assert isinstance(collapsed, StateVector)

    def test_entanglement_matrix(self):
        """Test EntanglementMatrix"""
        matrix = EntanglementMatrix(4)

        matrix.set_entanglement(0, 1, 0.8 + 0.2j)
        matrix.set_entanglement(1, 2, 0.7 + 0.3j)
        matrix.set_entanglement(0, 2, 0.6 + 0.4j)

        # Check symmetry
        assert matrix.get_entanglement(0, 1) == 0.8 + 0.2j
        assert matrix.get_entanglement(1, 0) == 0.8 - 0.2j

        # Clustering coefficient
        cluster_coef = matrix.cluster_coefficient(0)
        assert 0 <= cluster_coef <= 1

        # Find clusters
        clusters = matrix.find_clusters(threshold=0.5)
        assert len(clusters) >= 1

    def test_coherence_metric(self):
        """Test CoherenceMetric"""
        metric = CoherenceMetric()

        state = StateVector(np.array([1 / np.sqrt(2), 1 / np.sqrt(2)]))

        # Measure coherence
        coherence = metric.measure_state_coherence(state)
        assert coherence > 0

        # Track over time
        for t in range(10):
            metric.track_coherence(coherence * np.exp(-0.1 * t), t)

        # Estimate decoherence rate
        rate = metric.decoherence_rate()
        assert rate >= 0


class TestMindFractalIntegration:
    """Test QWAMOS integration with MindFractal Lab"""

    @pytest.mark.asyncio
    async def test_mindfractal_qwamos_initialization(self):
        """Test MindFractalQWAMOS initialization"""
        qwamos = MindFractalQWAMOS()

        await qwamos.initialize()

        assert qwamos.agents_initialized
        assert qwamos.protocols_initialized
        assert len(qwamos.engine.agents) == 8

    @pytest.mark.asyncio
    async def test_consciousness_model_creation(self):
        """Test creating consciousness model with QWAMOS"""
        qwamos = MindFractalQWAMOS()

        parameters = {"dimensions": 10, "nonlinearity": "tanh", "fractal_depth": 3}

        # Mock the engine cycle to prevent actual async execution
        with patch.object(qwamos, "_run_engine_cycle", return_value=asyncio.sleep(0)):
            with patch.object(
                qwamos,
                "_collect_results",
                return_value={
                    "task1": {"design": "quantum_fractal_architecture"},
                    "task2": {"mathematical": "nonlinear_dynamics"},
                    "task3": {"implementation": "optimized_algorithms"},
                    "task4": {"analysis": "fractal_dimension_2.37"},
                },
            ):
                result = await qwamos.create_consciousness_model(parameters)

        assert result["model_type"] == "quantum_fractal_consciousness"
        assert "quantum_properties" in result
        assert result["quantum_properties"]["superposition_utilized"]

    @pytest.mark.asyncio
    async def test_task_orchestration(self):
        """Test general task orchestration"""
        qwamos = MindFractalQWAMOS()
        await qwamos.initialize()  # Initialize first to create consensus

        task = "Analyze the fractal properties of consciousness emergence"

        # Mock consensus achievement
        with patch.object(
            qwamos.consensus,
            "achieve_consensus",
            return_value=(True, {"decision": {}, "entanglement": 0.8}),
        ):
            with patch.object(
                qwamos, "_run_engine_cycle", return_value=asyncio.sleep(0)
            ):
                result = await qwamos.orchestrate_research_task(task)

        assert result["status"] in ["completed", "partial"]
        assert "task_id" in result
        assert result["quantum_advantage_utilized"]

    def test_agent_determination(self):
        """Test determining primary agents for tasks"""
        qwamos = MindFractalQWAMOS()

        # Test various task descriptions
        design_agents = qwamos._determine_primary_agents(
            "Design a new consciousness architecture"
        )
        assert "Q0" in design_agents  # Architect
        assert "Q5" in design_agents  # Integrator

        math_agents = qwamos._determine_primary_agents(
            "Formalize mathematical framework"
        )
        assert "Q1" in math_agents  # Mathematician

        viz_agents = qwamos._determine_primary_agents("Visualize fractal dynamics")
        assert "Q4" in viz_agents  # Visualization Artist


class TestSpecializedAgents:
    """Test individual specialized agent behaviors"""

    @pytest.mark.asyncio
    async def test_meta_architect_processing(self):
        """Test Q0 Meta-Architect agent"""
        agent = Q0_MetaArchitect()

        task = QuantumTask(description="Design quantum consciousness architecture")
        result = await agent.process_task(task)

        assert result["agent"] == "Meta-Architect"
        assert result["status"] in ["processing", "completed"]

    @pytest.mark.asyncio
    async def test_mathematician_formalization(self):
        """Test Q1 Mathematical Formalist agent"""
        agent = Q1_MathematicalFormalist()

        task = QuantumTask(description="Formalize fractal dynamics equations")
        result = await agent.process_task(task)

        assert result["agent"] == "Mathematical-Formalist"
        if result["status"] == "completed":
            assert "formalization" in result or "proof" in result

    @pytest.mark.asyncio
    async def test_consciousness_modeler(self):
        """Test Q6 Consciousness Modeler agent"""
        agent = Q6_ConsciousnessModeler()

        task = QuantumTask(description="Model consciousness emergence patterns")
        result = await agent.process_task(task)

        assert result["agent"] == "Consciousness-Modeler"

    @pytest.mark.asyncio
    async def test_fractal_analyst(self):
        """Test Q7 Fractal Analyst agent"""
        agent = Q7_FractalAnalyst()

        task = QuantumTask(description="Analyze fractal dimension of attractor")
        result = await agent.process_task(task)

        assert result["agent"] == "Fractal-Analyst"


# Performance and stress tests
class TestPerformance:
    """Test performance and scalability"""

    @pytest.mark.asyncio
    async def test_multiple_task_handling(self):
        """Test handling multiple tasks concurrently"""
        engine = QWAMOSEngine()

        # Register all agents
        agents = [
            Q0_MetaArchitect(),
            Q1_MathematicalFormalist(),
            Q2_ComputationalEngineer(),
            Q3_DocumentationWeaver(),
        ]

        for agent in agents:
            engine.register_agent(agent)

        # Submit multiple tasks
        task_ids = []
        for i in range(10):
            task = QuantumTask(description=f"Task {i}", priority=np.random.rand())
            task_id = await engine.submit_task(task)
            task_ids.append(task_id)

        assert len(engine.tasks) == 10

    @pytest.mark.asyncio
    async def test_coherence_maintenance(self):
        """Test system coherence maintenance over time"""
        protocol = CoherenceProtocol()

        agents = {f"Q{i}": Mock(id=f"Q{i}", coherence=0.9 - i * 0.1) for i in range(5)}

        initial_coherence = await protocol.monitor_coherence(agents)

        # Apply error correction
        for agent in agents.values():
            agent.state_vector = np.array([1, 0])
            await protocol.apply_error_correction(agent)

        # Inject fresh coherence
        await protocol.inject_fresh_coherence(agents)

        final_coherence = await protocol.monitor_coherence(agents)

        # Coherence should be maintained or improved
        assert final_coherence >= initial_coherence * 0.8
