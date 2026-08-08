# Multi-Agent Orchestrator

Lightweight framework for multi-agent systems with MCP tool integration, task decomposition, sub-agent delegation, and failure recovery.

Built from patterns used in production platforms processing 10K+ monthly transactions across banking and enterprise workflows.

## Features

- **Task decomposition** with dependency tracking
- **Agent registry** with intent-based routing
- **MCP tool integration** for external APIs and databases
- **Failure recovery**: retry, fallback agents, human-in-the-loop escalation
- **Execution traces** with token, latency, and cost tracking

## Quick start

```bash
pip install -r requirements.txt
python src/demo.py
```

## Tech stack

Python, LangChain, LangGraph, OpenAI, Pydantic
