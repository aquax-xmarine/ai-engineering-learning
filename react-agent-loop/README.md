# ReAct Agent from Scratch

A minimal ReAct (Reasoning + Acting) agent in plain Python, with no frameworks, built to understand how the agent loop works.

The agent answers questions by reasoning step by step and choosing between two tools:
- `search_notes`: looks up facts in a small set of personal notes
- `calculator`: evaluates maths expressions

## How it works

Each step follows the Thought → Action → Observation loop from the ReAct paper ([Yao et al., 2022](https://arxiv.org/abs/2210.03629)):

```
Thought: I need to find out how many hours I work in a week.
Action: search_notes[internship]
Observation: My internship is 40 hours per week.
Thought: Now I need to find out how many hours I work in 12 weeks.
Action: calculator[40*12]
Observation: 480
Final Answer: You will work 480 hours in 12 weeks.
```

The conversation history is kept as a list of system, user, and assistant messages, so the model continues from earlier steps instead of restarting.

## Setup

1. Install [Ollama](https://ollama.com) and pull the model:
   ```bash
   ollama pull llama3.1
   ```
2. Install the Python package:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the agent:
   ```bash
   python3 agent.py
   ```

## What I learned

Building this taught me how small details break agents: models copy prompt examples literally, guess facts instead of searching, loop on ambiguous data, and restart if the conversation history isn't structured with proper message roles.