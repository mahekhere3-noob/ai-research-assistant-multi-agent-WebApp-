# 📚 AI Research Assistant-Web App

A web app where four AI agents collaborate to research, summarize,
fact-check, and write a structured report in response to your
question — built on OpenAI Swarm and Streamlit.

## Pipeline

1. **Research Agent** — gathers facts and context on your question
2. **Summarizer Agent** — condenses findings into key bullet points
3. **Fact Checker Agent** — flags unsupported or contradictory claims
4. **Report Generator Agent** — writes the final structured Markdown report

Each agent hands off to the next automatically.

## Setup — free option (recommended)

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Get a **free** API key at [console.groq.com](https://console.groq.com) —
   no credit card required. Sign up, then create an API key.
3. Create a `.env` file in this folder:
   ```
   GROQ_API_KEY=your_key_here
   ```
4. Run:
   ```bash
   streamlit run app.py
   ```

This runs the whole pipeline on Groq's free tier, using Llama 3.3 70B.
Free tier limits (subject to change on Groq's side): roughly 30
requests/minute and ~1,000 requests/day — plenty for personal/student
use. No cost, no card on file.

## Setup — paid option (OpenAI)

If you'd rather use OpenAI's gpt-4o-mini instead:

1. Get a key from [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
   (requires billing set up).
2. In `.env`, use:
   ```
   OPENAI_API_KEY=your_key_here
   ```
3. Run the same way. If both `GROQ_API_KEY` and `OPENAI_API_KEY` are
   present in `.env`, Groq is used.

## Note on Swarm

This project uses OpenAI's Swarm framework, which OpenAI has since
marked as deprecated in favor of their newer Agents SDK. Swarm still
works and is widely used for learning multi-agent handoff patterns
because it's small and readable, which is why it's used here. Swarm
is just a thin wrapper around a standard OpenAI-compatible client,
which is what makes pointing it at Groq's free endpoint possible.

## Security

- `.env` (containing your API key) is excluded via `.gitignore` —
  never commit it to a public repo.

## Tech stack

- Python, Streamlit
- OpenAI Swarm
- Groq (free) or OpenAI (paid) — both via an OpenAI-compatible API
