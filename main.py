import sys
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# NEW IMPORT PATH
from langchain_core.messages import SystemMessage, HumanMessage

# ---------------------------
# 1. Model Setup
# ---------------------------
load_dotenv()
model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.1,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    google_api_key=os.getenv("GEMINI_KEY"),
)

# ---------------------------
# 2. System Prompt
# ---------------------------

SYSTEM_PROMPT = """
You are a simple planning assistant.
Your job is to:
- Understand the user goal
- Break it into 4-5 steps
- Respond in clear bullet points
- Keep it concise
"""

# ---------------------------
# 3. Read CLI Input
# ---------------------------

if len(sys.argv) < 2:
    print("Usage: python planner_cli.py <your goal>")
    sys.exit(1)

user_input = " ".join(sys.argv[1:])

# ---------------------------
# 4. Call LLM
# ---------------------------

response = model.invoke([
    SystemMessage(content=SYSTEM_PROMPT),
    HumanMessage(content=user_input)
])

# ---------------------------
# 5. Output
# ---------------------------

print("\n===== PLAN =====\n")
print(response.content)
