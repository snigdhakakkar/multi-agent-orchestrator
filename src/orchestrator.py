"""Multi-agent orchestrator with task decomposition and failure recovery."""
import time, uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Callable
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

class TaskStatus(str, Enum):
    PENDING="pending"; RUNNING="running"; COMPLETED="completed"; FAILED="failed"; ESCALATED="escalated"

@dataclass
class Task:
    id: str; description: str; assigned_agent: Optional[str]=None
    status: TaskStatus=TaskStatus.PENDING; result: Optional[str]=None
    dependencies: list[str]=field(default_factory=list); retries: int=0; max_retries: int=2

@dataclass
class ExecutionTrace:
    task_id: str; agent_name: str; action: str; duration_ms: float; success: bool=True; error: Optional[str]=None

class BaseAgent:
    def __init__(self, name, model="gpt-4o", description=""):
        self.name, self.description = name, description
        self.llm = ChatOpenAI(model=model, temperature=0)
    def execute(self, task, context): raise NotImplementedError

class ReasoningAgent(BaseAgent):
    def execute(self, task, context):
        prior = "\n".join(f"- {k}: {v}" for k,v in context.get("prior_results",{}).items())
        return self.llm.invoke(f"Task: {task.description}\nPrior results:\n{prior or 'None'}\nProvide a clear answer.").content

class ToolAgent(BaseAgent):
    def __init__(self, name, tools, **kw):
        super().__init__(name, **kw)
        self.tools = {t.__name__: t for t in tools}
    def execute(self, task, context):
        descs = "\n".join(f"- {n}: {f.__doc__ or 'N/A'}" for n,f in self.tools.items())
        resp = self.llm.invoke(f"Tools:\n{descs}\nTask: {task.description}\nFormat: TOOL|args").content
        parts = resp.strip().split("|")
        if parts[0].strip() in self.tools:
            args = parts[1].split(",") if len(parts)>1 else []
            return str(self.tools[parts[0].strip()](*[a.strip() for a in args]))
        return "Tool not found"

class AgentOrchestrator:
    def __init__(self, model="gpt-4o", max_retries=2):
        self.llm = ChatOpenAI(model=model, temperature=0)
        self.agents = {}; self.max_retries = max_retries

    def register_agent(self, name, agent):
        agent.name = name; self.agents[name] = agent

    def decompose(self, query):
        agents_desc = "\n".join(f"- {n}: {a.description}" for n,a in self.agents.items())
        resp = self.llm.invoke(f"Decompose into sub-tasks.\nAgents:\n{agents_desc}\nRequest: {query}\nFormat per line: AGENT|description|deps or NONE")
        tasks = []
        for i, line in enumerate(resp.content.strip().split("\n")):
            if "|" not in line: continue
            p = [x.strip() for x in line.split("|")]
            if len(p)<2: continue
            deps = [d.strip() for d in p[2].split(",")] if len(p)>2 and p[2].upper()!="NONE" else []
            tasks.append(Task(id=f"task_{i+1}", description=p[1], assigned_agent=p[0] if p[0] in self.agents else None, dependencies=deps))
        return tasks

    def run(self, query):
        start = time.time()
        tasks = self.decompose(query)
        traces, results, completed = [], {}, set()
        while len(completed) < len(tasks):
            progress = False
            for t in tasks:
                if t.id in completed or not all(d in completed for d in t.dependencies): continue
                agent = self.agents.get(t.assigned_agent) or list(self.agents.values())[0]
                ts = time.time()
                try:
                    t.result = agent.execute(t, {"prior_results": results, "query": query})
                    t.status = TaskStatus.COMPLETED; results[t.id] = t.result
                    traces.append(ExecutionTrace(t.id, agent.name, "execute", (time.time()-ts)*1000))
                except Exception as e:
                    t.retries += 1
                    if t.retries >= t.max_retries:
                        t.status = TaskStatus.ESCALATED; t.result = f"ESCALATED: {e}"; results[t.id] = t.result
                        traces.append(ExecutionTrace(t.id, agent.name, "escalate", (time.time()-ts)*1000, False, str(e)))
                    else: continue
                completed.add(t.id); progress = True
            if not progress: break
        synthesis = "\n".join(f"- {t.description}: {t.result}" for t in tasks)
        answer = self.llm.invoke(f"Request: {query}\nResults:\n{synthesis}\nSynthesize final answer.").content
        return {"answer": answer, "tasks": tasks, "traces": traces, "latency_ms": (time.time()-start)*1000}

if __name__ == "__main__":
    o = AgentOrchestrator()
    o.register_agent("analyst", ReasoningAgent("analyst", description="Analyzes data and provides insights"))
    o.register_agent("planner", ReasoningAgent("planner", description="Creates action plans"))
    r = o.run("Analyze weather impact on Q3 scheduling and recommend mitigations")
    print(f"Answer: {r['answer']}\nTasks: {len(r['tasks'])}\nLatency: {r['latency_ms']:.0f}ms")
