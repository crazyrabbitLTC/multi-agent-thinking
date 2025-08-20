/**
 * Planner → Solvers → Judge: multi‑LLM reasoning blueprint (TypeScript)
 * ---------------------------------------------------------------------
 * This single‑file sketch shows a buildable multi‑agent loop using the Vercel AI SDK.
 * Swap model providers freely; OpenAI is shown by default. Keep loops bounded.
 *
 * Packages you'll likely want:
 *   npm i ai @ai-sdk/openai zod
 * Optional (for tools/tests/RAG):
 *   npm i jsdom undici cheerio sympy-wasm (or mathjs) neo4j-driver (for GraphRAG),
 *   @google-cloud/langchain (if you prefer LangChain) — not required here.
 *
 * Notes
 * - EvidenceLog records every step (inputs, outputs, citations, tests) for auditability.
 * - Judge should ideally be a *different* model than Planner/Solvers.
 * - Retriever here is a placeholder for GraphRAG: return nodes/edges/sources.
 */

import { z } from 'zod';
import { generateText } from 'ai';
import { createOpenAI } from '@ai-sdk/openai';
import { createGroq } from '@ai-sdk/groq';

// ────────────────────────────────────────────────────────────────────────────────
// Utility functions for timing and progress feedback
// ────────────────────────────────────────────────────────────────────────────────

function formatTime(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

function logProgress(message: string, indent = 0) {
  const prefix = '  '.repeat(indent);
  console.log(`${prefix}🔄 ${message}`);
}

function logSuccess(message: string, indent = 0) {
  const prefix = '  '.repeat(indent);
  console.log(`${prefix}✅ ${message}`);
}

function logError(message: string, indent = 0) {
  const prefix = '  '.repeat(indent);
  console.log(`${prefix}❌ ${message}`);
}

// Environment validation
function validateEnvironment(provider: ModelProvider): boolean {
  console.log(`🔧 Checking environment for ${provider.toUpperCase()}...`);
  
  switch (provider) {
    case 'openai': {
      if (!process.env.OPENAI_API_KEY) {
        logError('OPENAI_API_KEY environment variable is not set!');
        console.log('   Please set your API key: export OPENAI_API_KEY=your_key_here');
        return false;
      }
      
      if (process.env.OPENAI_API_KEY.length < 20) {
        logError('OPENAI_API_KEY appears to be invalid (too short)');
        return false;
      }
      break;
    }
    case 'groq': {
      if (!process.env.GROQ_API_KEY) {
        logError('GROQ_API_KEY environment variable is not set!');
        console.log('   Please set your API key: export GROQ_API_KEY=your_key_here');
        return false;
      }
      
      if (process.env.GROQ_API_KEY.length < 20) {
        logError('GROQ_API_KEY appears to be invalid (too short)');
        return false;
      }
      break;
    }
  }
  
  logSuccess(`Environment validated successfully for ${provider.toUpperCase()}`);
  return true;
}

// ────────────────────────────────────────────────────────────────────────────────
// Configure models (OpenAI and Groq providers)
// ────────────────────────────────────────────────────────────────────────────────

type ModelProvider = 'openai' | 'groq';

function createModels(provider: ModelProvider) {
  switch (provider) {
    case 'openai': {
      const openai = createOpenAI({ apiKey: process.env.OPENAI_API_KEY! });
      return {
        planner: openai('gpt-4o-mini'),
        solver: openai('gpt-4o-mini'),
        judge: openai('gpt-4o'),
        provider: 'OpenAI' as const,
      };
    }
    case 'groq': {
      const groq = createGroq({ apiKey: process.env.GROQ_API_KEY! });
      return {
        planner: groq('llama-3.3-70b-versatile'),
        solver: groq('llama-3.3-70b-versatile'),
        judge: groq('llama-3.3-70b-versatile'),
        provider: 'Groq' as const,
      };
    }
    default:
      throw new Error(`Unsupported provider: ${provider}`);
  }
}

// Global models object - will be initialized in main function
let MODELS: ReturnType<typeof createModels>;

// ────────────────────────────────────────────────────────────────────────────────
// Core types
// ────────────────────────────────────────────────────────────────────────────────

type ID = string;

type SubtaskKind = 'research' | 'reason' | 'verify' | 'coding' | 'math' | 'synthesis' | 'general';

interface Subtask {
  id: ID;
  kind: SubtaskKind;
  prompt: string;
  deps?: ID[]; // dependencies by subtask id
}

interface Plan {
  subtasks: Subtask[];
}

interface EvidenceItem {
  stepId: string;
  role: 'planner' | 'solver' | 'judge' | 'tool';
  input: Record<string, any>;
  output: Record<string, any>;
  citations?: string[];
  tests?: Array<Record<string, any>>;
}

class EvidenceLog {
  private items: EvidenceItem[] = [];
  add(item: EvidenceItem) { this.items.push(item); }
  toJSON() { return JSON.stringify(this.items, null, 2); }
  get all() { return [...this.items]; }
}

// ────────────────────────────────────────────────────────────────────────────────
// Retriever (GraphRAG placeholder)
// Return an evidence graph {nodes, edges, sources}
// ────────────────────────────────────────────────────────────────────────────────

class Retriever {
  async retrieve(query: string) {
    // TODO: replace with your GraphRAG implementation and real sources/IDs
    return {
      nodes: [{ id: 'claim1', type: 'claim', text: `Evidence for: ${query}` }],
      edges: [],
      sources: ['https://example.com/source'],
    } as const;
  }
}

// ────────────────────────────────────────────────────────────────────────────────
// Tooling: put unit tests, schema checks, static analysis, math solvers, etc.
// ────────────────────────────────────────────────────────────────────────────────

class Tooling {
  async runTests(artifact: Record<string, any>) {
    // Stub — integrate real tests here
    return [{ name: 'schema_check', passed: true }];
  }
}

// ────────────────────────────────────────────────────────────────────────────────
// Planner: decomposes a goal into a small DAG of subtasks
// ────────────────────────────────────────────────────────────────────────────────

const PlanSchema = z.object({
  subtasks: z.array(z.object({
    id: z.string(),
    kind: z.enum(['research', 'reason', 'verify', 'coding', 'math', 'synthesis', 'general']),
    prompt: z.string(),
    deps: z.array(z.string()).optional(),
  })).min(1).max(8),
});

class Planner {
  async makePlan(userGoal: string): Promise<Plan> {
    logProgress('Creating execution plan...', 1);
    const system = `You are a planning assistant. Decompose the goal into 3-6 atomic subtasks forming a DAG. Return strict JSON.`;
    const { text } = await generateText({
      model: MODELS.planner,
      system,
      prompt: `Goal: ${userGoal}\nReturn JSON with { subtasks: [{id, kind, prompt, deps?}] }` ,
      temperature: 0.2,
    });

    // Guarded parse; on failure, fall back to a default 3-step plan
    try {
      const json = JSON.parse(text);
      const plan = PlanSchema.parse(json);
      logSuccess(`Plan created with ${plan.subtasks.length} subtasks`, 1);
      return plan;
    } catch {
      logProgress('Plan parsing failed, using fallback plan', 1);
      const fallbackPlan = {
        subtasks: [
          { id: 's1', kind: 'research', prompt: `Find key facts for: ${userGoal}` },
          { id: 's2', kind: 'reason', prompt: 'Synthesize a candidate answer from s1', deps: ['s1'] },
          { id: 's3', kind: 'verify', prompt: 'Check s2 against sources and tests', deps: ['s2'] },
        ],
      };
      logSuccess(`Fallback plan created with ${fallbackPlan.subtasks.length} subtasks`, 1);
      return fallbackPlan;
    }
  }
}

// ────────────────────────────────────────────────────────────────────────────────
// Solver: self‑consistency (k samples), retrieval‑augmented
// ────────────────────────────────────────────────────────────────────────────────

class Solver {
  constructor(private retriever: Retriever) {}

  async propose(subtask: Subtask, context: Record<string, any>, k = 3) {
    logProgress(`Generating ${k} proposals for ${subtask.kind} task: ${subtask.id}`, 2);
    const retrieval = await this.retriever.retrieve(subtask.prompt);
    const proposals: Array<{ text: string; citations: string[]; evidence: any }> = [];

    for (let i = 0; i < k; i++) {
      logProgress(`Proposal ${i + 1}/${k}`, 3);
      const system = `You are the ${subtask.kind} specialist. Use the evidence graph and be concise but precise. Cite sources explicitly.`;
      const { text } = await generateText({
        model: MODELS.solver,
        system,
        prompt: JSON.stringify({ subtask, context, evidence_graph: retrieval }),
        temperature: 0.6,
      });
      proposals.push({ text, citations: retrieval.sources, evidence: retrieval });
    }

    logSuccess(`Generated ${proposals.length} proposals for ${subtask.id}`, 2);
    return proposals;
  }

  vote(proposals: Array<{ text: string; citations: string[]; evidence: any }>) {
    // Minimal heuristic: prefer the longest common substring style (proxy for consensus)
    // For the sketch, return the first; replace with a judge‑assisted ranker.
    return proposals[0];
  }
}

// ────────────────────────────────────────────────────────────────────────────────
// Judge: checks artifacts against tests and citations
// ────────────────────────────────────────────────────────────────────────────────

class Judge {
  constructor(private tooling: Tooling) {}

  async inspect(subtask: Subtask, artifact: Record<string, any>) {
    logProgress(`Judging artifact for ${subtask.id}`, 2);
    const tests = await this.tooling.runTests(artifact);
    const system = 'You are a strict verifier. Check factuality vs citations and logical consistency.';
    const { text: critique } = await generateText({
      model: MODELS.judge,
      system,
      prompt: JSON.stringify({ subtask, artifact, tests }),
      temperature: 0,
    });

    const passed = tests.every(t => t.passed);
    if (passed) {
      logSuccess(`Judge approved artifact for ${subtask.id}`, 2);
    } else {
      logProgress(`Judge rejected artifact for ${subtask.id}, retry needed`, 2);
    }
    return { passed, info: { critique, tests } } as const;
  }
}

// ────────────────────────────────────────────────────────────────────────────────
// Orchestrator: executes plan with bounded retries, logs evidence
// ────────────────────────────────────────────────────────────────────────────────

class Orchestrator {
  private log = new EvidenceLog();
  constructor(
    private planner: Planner,
    private solver: Solver,
    private judge: Judge,
    private maxRetries = 2,
  ) {}

  async run(goal: string) {
    console.log('\n🚀 Starting multi-agent reasoning system...');
    const startTime = Date.now();
    
    const plan = await this.planner.makePlan(goal);
    const done = new Set<ID>();
    const artifacts: Record<ID, Record<string, any>> = {};

    const ready = () => plan.subtasks.filter(s => !done.has(s.id) && (s.deps ?? []).every(d => done.has(d)));

    console.log(`\n📋 Executing plan with ${plan.subtasks.length} subtasks...`);
    let taskCount = 0;
    
    while (done.size < plan.subtasks.length) {
      const batch = ready();
      if (batch.length === 0) throw new Error('Deadlock in plan dependencies');

      for (const sub of batch) {
        taskCount++;
        console.log(`\n[${taskCount}/${plan.subtasks.length}] Processing subtask: ${sub.id} (${sub.kind})`);
        const artifact = await this.executeSubtask(sub, artifacts);
        artifacts[sub.id] = artifact;
        done.add(sub.id);
        logSuccess(`Completed subtask ${sub.id}`, 1);
      }
    }

    const totalTime = Date.now() - startTime;
    const finalId = plan.subtasks.findLast?.(Boolean)?.id ?? plan.subtasks[plan.subtasks.length - 1].id;
    const final = artifacts[finalId];
    
    console.log(`\n✨ Multi-agent reasoning completed in ${formatTime(totalTime)}`);
    return { final, evidence: this.log.toJSON(), plan, executionTime: totalTime } as const;
  }

  private async executeSubtask(sub: Subtask, artifacts: Record<ID, Record<string, any>>) {
    const context = Object.fromEntries((sub.deps ?? []).map(d => [d, artifacts[d]]));
    const startTime = Date.now();

    let lastCandidate: Record<string, any> | undefined;
    for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
      logProgress(`Attempt ${attempt + 1}/${this.maxRetries + 1}`, 1);
      const proposals = await this.solver.propose(sub, context, sub.kind === 'verify' ? 2 : 3);
      const candidate = this.solver.vote(proposals);

      this.log.add({
        stepId: `${sub.id}:${attempt}`,
        role: 'solver',
        input: { subtask: sub, context },
        output: candidate,
        citations: candidate.citations,
      });

      const { passed, info } = await this.judge.inspect(sub, candidate);
      this.log.add({ stepId: `${sub.id}:${attempt}`, role: 'judge', input: { artifact: candidate }, output: info });

      if (passed) {
        const taskTime = Date.now() - startTime;
        logSuccess(`Task ${sub.id} completed in ${formatTime(taskTime)}`, 1);
        return candidate;
      }
      lastCandidate = { ...candidate, critique: info.critique };
    }

    const taskTime = Date.now() - startTime;
    logError(`Task ${sub.id} exhausted retries after ${formatTime(taskTime)}`, 1);
    return { ...lastCandidate, failed: true, note: 'Max retries exhausted; returning best‑effort candidate.' };
  }
}

// ────────────────────────────────────────────────────────────────────────────────
// CLI Interface
// ────────────────────────────────────────────────────────────────────────────────

function printUsage() {
  console.log(`\n🤖 Multi-Agent Reasoning CLI\n`);
  console.log('Usage: npx tsx multi-agent-reasoning.ts [OPTIONS] "<your goal here>"\n');
  console.log('Options:');
  console.log('  -m, --model PROVIDER    Choose AI provider: openai (default) or groq');
  console.log('  -h, --help              Show this help message\n');
  console.log('Examples:');
  console.log('  npx tsx multi-agent-reasoning.ts "Explain quantum computing basics"');
  console.log('  npx tsx multi-agent-reasoning.ts -m groq "How does blockchain work?"');
  console.log('  npx tsx multi-agent-reasoning.ts --model openai "Optimize React performance"\n');
  console.log('Environment Variables:');
  console.log('  OPENAI_API_KEY - Required for OpenAI provider');
  console.log('  GROQ_API_KEY   - Required for Groq provider');
  console.log('  SAVE_EVIDENCE=1 - Optional: Save detailed evidence log to file\n');
  console.log('Supported Models:');
  console.log('  OpenAI: GPT-4o, GPT-4o-mini');
  console.log('  Groq:   Llama 3.3 70B, Gemma 2 9B, Qwen QwQ 32B\n');
}

export async function runMultiAgentReasoning(goal: string, provider: ModelProvider = 'openai') {
  if (!validateEnvironment(provider)) {
    process.exit(1);
  }
  
  // Initialize models with selected provider
  MODELS = createModels(provider);
  console.log(`\n🤖 Using ${MODELS.provider} models`);

  const retriever = new Retriever();
  const tooling = new Tooling();
  const planner = new Planner();
  const solver = new Solver(retriever);
  const judge = new Judge(tooling);
  const orch = new Orchestrator(planner, solver, judge, 2);

  console.log(`\n🎯 GOAL: ${goal}`);
  const result = await orch.run(goal);
  
  console.log('\n' + '='.repeat(80));
  console.log('🏆 FINAL RESULT');
  console.log('='.repeat(80));
  console.log(JSON.stringify(result.final, null, 2));
  
  console.log('\n' + '='.repeat(80));
  console.log('📋 EXECUTION PLAN');
  console.log('='.repeat(80));
  result.plan.subtasks.forEach((task, i) => {
    console.log(`${i + 1}. [${task.kind.toUpperCase()}] ${task.prompt}`);
    if (task.deps) {
      console.log(`   Dependencies: ${task.deps.join(', ')}`);
    }
  });
  
  console.log(`\n⏱️  Total execution time: ${formatTime(result.executionTime)}`);
  
  // Optionally save detailed evidence log
  if (process.env.SAVE_EVIDENCE) {
    const fs = await import('fs');
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = `evidence-${timestamp}.json`;
    fs.writeFileSync(filename, result.evidence);
    console.log(`\n📝 Evidence log saved to: ${filename}`);
  }
  
  return result;
}

// Parse CLI arguments
function parseArgs(args: string[]) {
  const result = {
    provider: 'openai' as ModelProvider,
    goal: '',
    help: false,
  };
  
  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    
    switch (arg) {
      case '-h':
      case '--help':
        result.help = true;
        break;
      case '-m':
      case '--model':
        const nextArg = args[i + 1];
        if (nextArg === 'openai' || nextArg === 'groq') {
          result.provider = nextArg;
          i++; // Skip next argument
        } else {
          console.error(`Error: Invalid provider '${nextArg}'. Use 'openai' or 'groq'.`);
          process.exit(1);
        }
        break;
      default:
        if (!arg.startsWith('-') && !result.goal) {
          result.goal = arg;
        }
        break;
    }
  }
  
  return result;
}

// CLI entry point
if (import.meta.url === `file://${process.argv[1]}`) {
  const args = parseArgs(process.argv.slice(2));
  
  if (args.help || !args.goal) {
    printUsage();
    process.exit(args.help ? 0 : 1);
  }
  
  runMultiAgentReasoning(args.goal, args.provider).catch(err => {
    logError(`Execution failed: ${err.message}`);
    console.error(err);
    process.exit(1);
  });
}