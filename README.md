# Multi-Agent Reasoning CLI

A TypeScript implementation of a multi-agent reasoning system using the Planner → Solvers → Judge pattern with support for OpenAI and Groq models.

## Features

- 🧠 **Multi-Agent Architecture**: Planner decomposes goals, Solvers generate proposals, Judge validates results
- ⚡ **Multiple AI Providers**: Support for OpenAI and Groq models
- 🔄 **Real-time Progress**: Visual feedback with timing information
- 📊 **Evidence Logging**: Complete audit trail of reasoning process
- 🎯 **CLI Interface**: Simple command-line usage

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd multi-agent-reasoning

# Install dependencies
npm install

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

## Environment Variables

```bash
# Required: Choose one or both
OPENAI_API_KEY=your_openai_api_key_here
GROQ_API_KEY=your_groq_api_key_here

# Optional: Save detailed evidence logs
SAVE_EVIDENCE=1
```

## Usage

```bash
# Using OpenAI (default)
npx tsx multi-agent-reasoning.ts "Explain quantum computing basics"

# Using Groq
npx tsx multi-agent-reasoning.ts -m groq "How does blockchain work?"

# Get help
npx tsx multi-agent-reasoning.ts --help
```

## Example Output

```
🎯 GOAL: Explain the basics of machine learning

🚀 Starting multi-agent reasoning system...
  🔄 Creating execution plan...
  ✅ Plan created with 3 subtasks

📋 Executing plan with 3 subtasks...
...
✨ Multi-agent reasoning completed in 66.95s

🏆 FINAL RESULT
================================================================================
{
  "text": "Machine learning is a subset of artificial intelligence...",
  "citations": ["https://example.com/source"]
}
```

## Architecture

- **Planner**: Decomposes complex goals into atomic subtasks
- **Solver**: Generates multiple proposals using self-consistency
- **Judge**: Validates outputs against tests and citations
- **Orchestrator**: Manages execution flow and evidence logging

## Supported Models

### OpenAI
- GPT-4o, GPT-4o-mini for planning and solving
- Different models for judging to reduce bias

### Groq
- Llama 3.3 70B, Gemma 2 9B for fast inference
- Reasoning models like Qwen QwQ 32B

## Development

```bash
# Run with development logging
DEBUG=1 npx tsx multi-agent-reasoning.ts "your goal"

# Save evidence logs
SAVE_EVIDENCE=1 npx tsx multi-agent-reasoning.ts "your goal"
```

## License

MIT