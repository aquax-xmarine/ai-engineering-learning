import ollama
import re

NOTES = {
    "react": "ReAct stands for Reasoning + Acting, from a 2022 paper.",
    "internship": "My internship is 40 hours per week.",
    "books": "I am reading A Little Life, War and Peace, and The Secret History.",
}


SYSTEM_PROMPT = """
Answer the question using these tools:
- calculator[expression]: evaluates maths, e.g. calculator[23*47]
- search_notes[topic]: searches my personal notes, e.g. search_notes[books]

Rules:
- Never assume facts about me. Always use search_notes for personal information.
- Only use the calculator if the question needs maths.
- As soon as you can answer the question, write the Final Answer.

Always use this exact format:
Thought: what you should do next
Action: tool_name[input]
Observation: the tool result

Example 1:
Question: What book am I reading?
Thought: I need to check my notes about books.
Action: search_notes[books]
Observation: I am reading A Little Life.
Thought: The note answers the question directly.
Final Answer: You are reading A Little Life.

Example 2:
Question: How many pages do I read in 3 days?
Thought: I need to find how many pages I read per day.
Action: search_notes[reading]
Observation: I read 20 pages per day.
Thought: Now I multiply 20 by 3.
Action: calculator[20*3]
Observation: 20*3 = 60
Final Answer: You read 60 pages in 3 days.
"""

def call_llm(messages):
    response = ollama.chat(
        model="llama3.1",
        messages=messages,
        options={"stop": ["Observation:"]},
    )
    return response["message"]["content"]


def calculator(expression):
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"Error: {e}"

def search_notes(query):
    query = query.lower()
    for key, text in NOTES.items():
        if key in query or query in key:
            return text
    return "No matching note found."


def parse_action(text):
    match = re.search(r"Action:\s*(\w+)\[(.*?)\]", text)
    if match:
        return match.group(1), match.group(2)
    return None, None




TOOLS ={"calculator": calculator, "search_notes": search_notes}





def run_agent(question, max_steps=5):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}"},
    ]
    for step in range(max_steps):
        reply = call_llm(messages)
        messages.append({"role": "assistant", "content": reply})
        print(f"\n--- Step {step + 1} ---\n{reply}")

        if "Final Answer:" in reply:
            return reply.split("Final Answer:")[-1].strip()

        tool, arg = parse_action(reply)
        result = TOOLS[tool](arg) if tool in TOOLS else f"Unknown tool: {tool}."
        print(f"Observation: {result}")
        messages.append({"role": "user", "content": f"Observation: {result}"})

    return "Stopped: reached max steps."

print("\nANSWER:", run_agent("How many hours would I work in 12 weeks of my internship?"))






