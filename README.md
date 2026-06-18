# Reflective Plan-and-Execute Agent

A Python prototype for a plan-and-execute agent architecture. The current baseline
focuses on clear task planning, step execution, lightweight replanning, and final
answer synthesis.

## Current Features

- Generates a structured execution plan for a user task.
- Executes each plan step with access to prior step history.
- Uses a lightweight critic check to decide whether replanning is needed.
- Replans only the remaining work when execution diverges from expectations.
- Synthesizes all step results into a final structured answer.

## Roadmap

1. Clean baseline implementation.
2. Structured agent state and execution trace.
3. Task composition with goals, constraints, and success criteria.
4. Critic and reflexion loop for self-correction.
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
memory, critic-based self-correction, reflexion records, and search-based
reasoning over multiple candidate solution paths.
