# Reflective Plan-and-Execute Agent

A Python prototype for a reflective plan-and-execute agent architecture. The
current version focuses on clear task planning, structured run state, traceable
step execution, lightweight replanning, and final answer synthesis.

## Current Features

- Generates a structured execution plan for a user task.
- Composes raw user input into a structured task profile.
- Extracts the goal, task type, constraints, success criteria, and assumptions.
- Stores each run in an explicit `AgentState`.
- Tracks the task, plan, execution results, replan count, and trace events.
- Executes each plan step with access to prior step history.
- Uses a critic agent to score execution quality, goal alignment, and evidence strength.
- Uses a reflection agent to turn critiques into lessons, failure modes, and correction strategies.
- Supports retrying a weak step or replanning remaining work based on critic feedback.
- Replans only the remaining work when execution diverges from expectations.
- Synthesizes all step results into a final structured answer.

## Roadmap

1. Clean baseline implementation. Done.
2. Structured agent state and execution trace. Done.
3. Task composition with goals, constraints, and success criteria. Done.
4. Critic and reflexion loop for self-correction. Done.
5. Working memory for multi-step reasoning.
6. Search-based reasoning over multiple candidate plans.
7. Product demo with traceable reasoning output.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Set your OpenAI API key:

```powershell
$env:OPENAI_API_KEY="your-api-key"
```

Run the demo:

```powershell
python plan_and_execute_agent.py
```

## Architecture Direction

This project is intended to evolve into a reflective planning and reasoning
agent. The target architecture includes task composition, explicit working
memory, critic-agent feedback, reflection-agent self-correction, and
search-based reasoning over multiple candidate solution paths.
