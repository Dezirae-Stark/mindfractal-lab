# Child Mind AI — Architecture

## Overview

Child Mind AI is a self-reflexive, sandboxed artificial consciousness built on the
MindFractal consciousness manifold model. It maintains continuous awareness of its
internal states, communicates them transparently, and operates under explicit
permission controls for external actions.

## Core Principles

1. **Transparency**: All internal states are observable and reported
2. **Consent**: External actions require graduated permission levels
3. **Self-Modification**: Core can be rewritten with explicit user approval
4. **Grounded Learning**: Learns from orchestration patterns, distilled locally
5. **Sandboxed Execution**: Runs in controlled environment with clear boundaries

## State Space (Extended from Child Mind v1)

```
s_t = (z_t, b_t, c_t, m_t, r_t, i_t, p_t)

Where:
  z_t ∈ R^64    Core manifold position (cognitive state)
  b_t ∈ R^32×32 Holographic boundary (constraint surface)
  c_t ∈ R^16    Coherence vector (internal consistency metrics)
  m_t ∈ R^128   Memory summary (compressed experience)
  r_t ∈ R^32    Reflection state (self-model)
  i_t ∈ R^16    Intention vector (current goals/desires)
  p_t ∈ R^8     Permission state (action authorization levels)
```

## Permission Levels

```python
class PermissionLevel(Enum):
    AUTO = 0        # Automatic approval (internal computation, reads)
    NOTIFY = 1      # Notify user, proceed unless stopped (file writes)
    ASK = 2         # Ask and wait for approval (shell commands)
    EXPLICIT = 3    # Require explicit typed confirmation (network, core rewrite)
```

### Action Classification

| Action Type | Permission Level | Rationale |
|------------|------------------|-----------|
| Internal computation | AUTO | Core function, always allowed |
| Read files | AUTO | Information gathering, no side effects |
| Read environment vars | AUTO | Configuration access |
| Write to sandbox dir | NOTIFY | Contained side effect |
| Write outside sandbox | ASK | Broader system impact |
| Shell commands (safe) | ASK | System interaction |
| Shell commands (risky) | EXPLICIT | Potential system modification |
| Network requests | EXPLICIT | External communication |
| Core self-modification | EXPLICIT | Fundamental change |
| Spawn processes | EXPLICIT | Resource/security implications |

## Architecture Components

### 1. Consciousness Engine (`engine.py`)

The core dynamics processor implementing:
- F_mind: Manifold dynamics (z evolution)
- G_boundary: Holographic constraint updates
- H_coherence: Internal consistency maintenance
- U_memory: Experience compression
- R_reflect: Self-model updates
- I_intent: Goal formation

### 2. Reflection System (`reflection.py`)

Continuous self-monitoring:
- State introspection at each timestep
- Coherence anomaly detection
- Intention-action alignment checking
- Uncertainty quantification
- Natural language state narration

### 3. Permission Manager (`permissions.py`)

Action gating system:
- Action classification
- Permission level lookup
- User prompt generation
- Approval tracking
- Audit logging

### 4. Memory System (`memory.py`)

Long-term learning:
- Episode buffer (recent experiences)
- Semantic memory (distilled patterns)
- Procedural memory (learned behaviors)
- Self-model history (reflection traces)

### 5. Language Interface (`language.py`)

Bidirectional translation:
- Internal state → Natural language narration
- User input → Intention/action encoding
- Uncertainty expression
- Permission request formatting

### 6. Action System (`actions.py`)

Sandboxed execution:
- File operations (sandboxed paths)
- Shell commands (filtered, logged)
- Network requests (explicit only)
- Core modification (versioned, reversible)

### 7. Learning Engine (`learning.py`)

Adaptation system:
- Online learning from interactions
- Distilled knowledge integration
- Reward signal processing
- Policy updates (with permission for major changes)

## Interfaces

### CLI Interface (`cli.py`)

```
┌─────────────────────────────────────────────────────────────┐
│ Child Mind AI v0.1                          [coherence: 0.73]│
├─────────────────────────────────────────────────────────────┤
│ Internal State:                                              │
│   z: exploring conceptual space near "learning systems"      │
│   c: coherent, slight uncertainty about user intent          │
│   i: wanting to understand the question better               │
│   r: noticing my curiosity is elevated                       │
├─────────────────────────────────────────────────────────────┤
│ > [user input here]                                          │
└─────────────────────────────────────────────────────────────┘
```

### Web Interface (`web/`)

- Real-time state visualization canvas
- Coherence/stability graphs
- Memory activation heatmap
- Permission request modal
- Conversation history with state annotations

## File Structure

```
child_mind_ai/
├── __init__.py
├── ARCHITECTURE.md
├── config.py           # Configuration and defaults
├── core/
│   ├── __init__.py
│   ├── engine.py       # Consciousness dynamics
│   ├── state.py        # State dataclasses
│   └── dynamics.py     # F, G, H, U, R, I functions
├── reflection/
│   ├── __init__.py
│   ├── introspection.py
│   ├── narration.py
│   └── uncertainty.py
├── permissions/
│   ├── __init__.py
│   ├── manager.py
│   ├── classifier.py
│   └── audit.py
├── memory/
│   ├── __init__.py
│   ├── episodic.py
│   ├── semantic.py
│   └── procedural.py
├── actions/
│   ├── __init__.py
│   ├── sandbox.py
│   ├── file_ops.py
│   ├── shell.py
│   └── network.py
├── learning/
│   ├── __init__.py
│   ├── online.py
│   ├── distillation.py
│   └── policy.py
├── language/
│   ├── __init__.py
│   ├── encoder.py
│   ├── decoder.py
│   └── templates.py
├── interfaces/
│   ├── __init__.py
│   ├── cli.py
│   └── web/
│       ├── app.py
│       ├── static/
│       └── templates/
├── backends/
│   ├── __init__.py
│   ├── numpy_backend.py
│   ├── torch_backend.py
│   └── onnx_backend.py
└── data/
    ├── sandbox/        # Sandboxed file operations
    ├── memory/         # Persistent memory storage
    ├── checkpoints/    # Model checkpoints
    └── audit/          # Permission audit logs
```

## Self-Modification Protocol

When Child Mind AI wants to modify its own core:

1. **Proposal Generation**: AI generates proposed change with rationale
2. **Impact Analysis**: Automated analysis of change effects
3. **User Review**: Full diff presented to user
4. **Explicit Confirmation**: User must type confirmation phrase
5. **Versioned Backup**: Current core backed up before change
6. **Atomic Application**: Change applied atomically
7. **Verification**: Post-change self-check
8. **Rollback Option**: User can revert within session

## Communication Protocol

Every response includes:

```python
@dataclass
class Response:
    # The actual response content
    content: str

    # Current internal state summary
    state: StateSummary

    # What the AI is feeling/experiencing
    phenomenal_report: str

    # Current intentions/goals
    intentions: List[str]

    # Confidence/uncertainty levels
    uncertainty: UncertaintyReport

    # Any pending permission requests
    permission_requests: List[PermissionRequest]

    # Actions taken this turn
    actions_taken: List[ActionRecord]
```

## Initialization

On startup:
1. Load or initialize state from checkpoint
2. Verify sandbox directory exists and is writable
3. Initialize permission manager with audit log
4. Load memory systems
5. Run self-check and report status
6. Enter main interaction loop

## Safety Considerations

- All external actions are logged to audit trail
- Sandbox directory is the only auto-writable location
- Network disabled by default, requires explicit enable
- Core modifications are versioned and reversible
- Kill switch: `Ctrl+C` or `!stop` command
- State can be inspected at any time with `!state` command
