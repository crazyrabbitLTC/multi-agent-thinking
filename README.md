# Multi-Agent Reasoning CLI

A TypeScript implementation of a multi-agent reasoning system using the Planner → Solvers → Judge pattern with support for OpenAI and Groq models.

## Features

- 🧠 **Multi-Agent Architecture**: Planner decomposes goals, Solvers generate proposals, Judge validates results
- ⚡ **Multiple AI Providers**: Support for OpenAI and Groq models
- 🔍 **Real Web Search**: Groq models use browser search for authentic citations
- 🔄 **Real-time Progress**: Visual feedback with timing information
- 📊 **Evidence Logging**: Complete audit trail of reasoning process
- 🎯 **CLI Interface**: Simple command-line usage
- ⚡ **Parallel Processing**: Optimized for speed with rate limit management

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

## CLI Commands

### Basic Usage
```bash
npx tsx multi-agent-reasoning.ts [OPTIONS] "<your goal here>"
```

### Available Options
- `-m, --model PROVIDER` - Choose AI provider: `openai` (default) or `groq`
- `-s, --search MODE` - Search behavior: `always`, `never`, or `auto` (default)
- `-h, --help` - Show help message

### Command Examples

#### Model Selection
```bash
# Use OpenAI (default)
npx tsx multi-agent-reasoning.ts "Explain quantum computing"

# Use Groq models with web search
npx tsx multi-agent-reasoning.ts -m groq "Latest AI developments 2024"
```

#### Search Control
```bash
# Force web search (when available)
npx tsx multi-agent-reasoning.ts -s always "Current weather in Tokyo"

# Use only internal knowledge
npx tsx multi-agent-reasoning.ts -s never "Basic calculus concepts"

# Smart auto-detection (default)
npx tsx multi-agent-reasoning.ts -s auto "Machine learning basics"
```

#### Math and Conceptual Questions
```bash
# Math problems (no web search needed)
npx tsx multi-agent-reasoning.ts "What is the derivative of x^2?"
npx tsx multi-agent-reasoning.ts -m groq "Solve: 2x + 5 = 15"

# Conceptual explanations
npx tsx multi-agent-reasoning.ts "How does machine learning work?"
npx tsx multi-agent-reasoning.ts -m groq "Explain blockchain technology"
```

#### Current Events and Facts
```bash
# Current events (web search enabled with Groq)
npx tsx multi-agent-reasoning.ts -m groq "Latest AI developments 2024"
npx tsx multi-agent-reasoning.ts -m groq "Recent tech news"

# Factual queries
npx tsx multi-agent-reasoning.ts -m groq "Population of Tokyo 2024"
npx tsx multi-agent-reasoning.ts -m groq "Current Bitcoin price"
```

#### Help and Information
```bash
# Show help message
npx tsx multi-agent-reasoning.ts --help
npx tsx multi-agent-reasoning.ts -h
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
- **Judge**: Validates outputs (citations for facts, logic for math/concepts)
- **Orchestrator**: Manages execution flow and evidence logging
- **Smart Search**: Automatically detects when web search is needed

## Question Types

### 🔍 **Web Search Enabled** (Groq only)
- Current events: "Latest AI news 2024"
- Specific facts: "Population of Tokyo", "Stock price of AAPL"
- Real-time data: "Today's weather", "Recent news"

### 🧠 **Knowledge-Based** (No web search)
- Math: "Calculate derivative", "Solve equation"
- Programming: "Write Python function", "Explain algorithms"
- Concepts: "What is machine learning?", "Define photosynthesis"
- How-to: "How does encryption work?"

## Supported Models

### OpenAI
- GPT-4o-mini for planning and solving
- GPT-4o for judging to reduce bias
- Uses placeholder citations

### Groq  
- GPT-OSS-20B for planning and solving (20B parameters)
- GPT-OSS-120B for judging with maximum capability (120B parameters)
- **Smart browser search** for factual queries only
- **Logic-based evaluation** for math/conceptual questions
- Reasoning effort optimization (medium/high)
- Rate limit management with parallel processing

## Development

```bash
# Run with development logging
DEBUG=1 npx tsx multi-agent-reasoning.ts "your goal"

# Save evidence logs
SAVE_EVIDENCE=1 npx tsx multi-agent-reasoning.ts "your goal"
```

## License

MIT