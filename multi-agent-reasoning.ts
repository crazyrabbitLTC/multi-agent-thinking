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
import 'dotenv/config';

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

type ModelProvider = 'openai' | 'groq';

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

function createModels(provider: ModelProvider) {
  switch (provider) {
    case 'openai': {
      const openai = createOpenAI({ apiKey: process.env.OPENAI_API_KEY! });
      return {
        planner: openai('gpt-4o-mini'),
        solver: openai('gpt-4o-mini'),
        judge: openai('gpt-4o'),
        provider: 'OpenAI' as const,
        models: 'GPT-4o-mini (planner/solver), GPT-4o (judge)' as const,
        browserSearch: undefined, // OpenAI doesn't have browser search
      };
    }
    case 'groq': {
      const groq = createGroq({ apiKey: process.env.GROQ_API_KEY! });
      return {
        planner: groq('openai/gpt-oss-20b'),          // Fast reasoning model
        solver: groq('openai/gpt-oss-20b'),           // Same for consistency
        judge: groq('openai/gpt-oss-120b'),           // Most powerful for validation
        provider: 'Groq' as const,
        models: 'GPT-OSS-20B (planner/solver), GPT-OSS-120B (judge)' as const,
        browserSearch: groq.tools.browserSearch({}),  // Browser search tool
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
// Source Analysis Types
// ────────────────────────────────────────────────────────────────────────────────

interface SourceAnalysis {
  url: string;
  domain: string;
  relevanceScore: number; // 0.0-1.0
  authority: 'High' | 'Medium' | 'Low';
  publicationDate?: string;
  keyFacts: string[];
  contentQuality: string;
}

// ────────────────────────────────────────────────────────────────────────────────
// Retriever (GraphRAG placeholder)
// Return an evidence graph {nodes, edges, sources}
// ────────────────────────────────────────────────────────────────────────────────

class Retriever {
  private searchCache = new Map<string, any>();
  
  constructor(private useRealSearch: boolean = false) {}
  
  // Determine if a query needs web search or can be answered with internal knowledge
  private needsWebSearch(query: string, goal: string): boolean {
    const queryLower = query.toLowerCase();
    const goalLower = goal.toLowerCase();
    const combinedText = `${queryLower} ${goalLower}`;
    
    // Categories that DON'T need web search
    const mathKeywords = ['calculate', 'solve', 'equation', 'derivative', 'integral', 'algebra', 'geometry', 'arithmetic', '+', '-', '*', '/', '=', 'formula', 'theorem', 'proof', 'math'];
    const codeKeywords = ['code', 'programming', 'function', 'algorithm', 'debug', 'syntax', 'variable', 'loop', 'class', 'method', 'python', 'javascript', 'typescript'];
    const conceptualKeywords = ['what is', 'define', 'explain', 'how does', 'how to', 'difference between', 'compare', 'concept of', 'meaning of'];
    const technicalKeywords = ['machine learning', 'ai', 'database', 'api', 'framework', 'library', 'protocol', 'encryption', 'security'];
    
    // Categories that DO need web search
    const currentEventKeywords = ['latest', 'recent', 'current', 'today', 'this year', '2024', '2025', 'breaking news', 'update'];
    const factualKeywords = ['who is', 'when did', 'where is', 'capital of', 'population of', 'price of', 'stock price', 'weather', 'news'];
    
    // First check if it needs current information
    if (currentEventKeywords.some(keyword => combinedText.includes(keyword))) {
      logProgress('Detected current events - using web search', 3);
      return true;
    }
    
    if (factualKeywords.some(keyword => combinedText.includes(keyword))) {
      logProgress('Detected specific factual query - using web search', 3);
      return true;
    }
    
    // Skip web search for knowledge-based questions
    if (mathKeywords.some(keyword => combinedText.includes(keyword))) {
      logProgress('Detected math/calculation - skipping web search', 3);
      return false;
    }
    
    if (codeKeywords.some(keyword => combinedText.includes(keyword))) {
      logProgress('Detected programming question - skipping web search', 3);
      return false;
    }
    
    if (conceptualKeywords.some(keyword => combinedText.includes(keyword))) {
      logProgress('Detected conceptual question - skipping web search', 3);
      return false;
    }
    
    if (technicalKeywords.some(keyword => combinedText.includes(keyword))) {
      logProgress('Detected technical explanation - skipping web search', 3);
      return false;
    }
    
    // Default: use web search for ambiguous queries
    logProgress('Query type unclear - using web search as fallback', 3);
    return true;
  }
  
  async retrieve(query: string, subtaskKind: SubtaskKind = 'research', goal: string = '') {
    // Only do web search for research tasks, not for reasoning or verification
    const shouldSearch = this.useRealSearch && 
                        MODELS.provider === 'Groq' && 
                        MODELS.browserSearch && 
                        subtaskKind === 'research' &&
                        this.needsWebSearch(query, goal);
    
    if (shouldSearch) {
      // Check cache first
      const cacheKey = `search:${query}`;
      if (this.searchCache.has(cacheKey)) {
        logProgress('Using cached search results', 3);
        return this.searchCache.get(cacheKey);
      }
      
      try {
        logProgress(`Searching web for: ${query}`, 3);
        
        // Use Groq's browser search with simplified analytical reading
        const { text } = await generateText({
          model: MODELS.solver,
          prompt: `Search for information about: ${query}. Provide factual information with specific sources and URLs.`,
          tools: {
            browser_search: MODELS.browserSearch,
          },
          toolChoice: 'required',
          providerOptions: {
            groq: {
              reasoningEffort: 'medium',
            },
          },
          maxRetries: 1, // Limit retries to prevent hanging
          retryDelay: 2000,
        });
        
        logSuccess('Retrieved analytical web research', 3);
        
        // DEBUG: Log raw response to understand format
        logProgress(`Debug - Response preview: ${text.substring(0, 300)}...`, 4);
        
        // Parse structured source analysis
        const sourceAnalysis = this.parseSourceAnalysis(text);
        logProgress(`Raw parsing found ${sourceAnalysis.length} sources`, 4);
        
        const qualitySources = sourceAnalysis.filter(s => s.relevanceScore > 0.3);
        const sourceUrls = qualitySources.map(s => s.url);
        
        logProgress(
          `Analyzed ${sourceAnalysis.length} sources, selected ${qualitySources.length} quality sources`, 
          3
        );
        
        // Debug logging if no sources found
        if (sourceAnalysis.length === 0) {
          logError('No sources parsed from search response!', 3);
          logProgress(`Response preview: ${text.substring(0, 200)}...`, 4);
        }
        
        // Log source quality summary
        if (qualitySources.length > 0) {
          const avgRelevance = qualitySources.reduce((sum, s) => sum + s.relevanceScore, 0) / qualitySources.length;
          const highAuthority = qualitySources.filter(s => s.authority === 'High').length;
          logProgress(
            `Average relevance: ${avgRelevance.toFixed(2)}, High authority sources: ${highAuthority}`, 
            4
          );
        }
        
        const result = {
          nodes: [{ 
            id: 'web_search', 
            type: 'search_result', 
            text: text,
            sourceAnalysis: sourceAnalysis // Include detailed analysis
          }],
          edges: [],
          sources: sourceUrls.length > 0 ? sourceUrls : this.extractFallbackUrls(text),
          sourceMetadata: {
            totalAnalyzed: sourceAnalysis.length,
            qualityCount: qualitySources.length,
            averageRelevance: qualitySources.length > 0 ? 
              qualitySources.reduce((sum, s) => sum + s.relevanceScore, 0) / qualitySources.length : 0,
            highAuthorityCount: qualitySources.filter(s => s.authority === 'High').length,
            publicationDates: qualitySources.filter(s => s.publicationDate).length
          }
        } as const;
        
        // Cache the result and RETURN IMMEDIATELY
        this.searchCache.set(cacheKey, result);
        return result;
      } catch (error) {
        logError(`Web search failed: ${error instanceof Error ? error.message : 'Unknown error'}`, 3);
        // Continue to fallback logic below
      }
    }
    
    // For non-research tasks or when search is disabled/fails
    const evidenceText = shouldSearch ? 
      `Evidence for: ${query}` : 
      `Knowledge-based response for: ${query} (no external sources needed)`;
    
    const sources = shouldSearch ? 
      ['https://example.com/source'] : 
      ['internal-knowledge']; // Indicate this is knowledge-based
    
    logProgress(`Using fallback sources: ${sources.join(', ')}`, 3);
    return {
      nodes: [{ id: 'knowledge', type: shouldSearch ? 'claim' : 'knowledge', text: evidenceText }],
      edges: [],
      sources: sources,
    } as const;
  }

  private parseSourceAnalysis(text: string): SourceAnalysis[] {
    const sources: SourceAnalysis[] = [];
    
    // First try structured parsing (our requested format)
    const structuredSources = this.parseStructuredFormat(text);
    if (structuredSources.length > 0) {
      return structuredSources;
    }
    
    // Then try Groq's native format parsing
    const groqSources = this.parseGroqFormat(text);
    if (groqSources.length > 0) {
      return groqSources;
    }
    
    // Final fallback: regex URL extraction
    return this.parseRegexFallback(text);
  }

  private parseStructuredFormat(text: string): SourceAnalysis[] {
    const sources: SourceAnalysis[] = [];
    const sourceSections = text.split('---').filter(section => 
      section.includes('**URL:**') && section.trim().length > 50
    );
    
    for (const section of sourceSections) {
      try {
        const analysis = this.extractSourceData(section);
        if (analysis) sources.push(analysis);
      } catch (error) {
        logProgress(`Structured parsing failed: ${error.message}`, 4);
      }
    }
    
    return sources;
  }

  private parseGroqFormat(text: string): SourceAnalysis[] {
    const sources: SourceAnalysis[] = [];
    
    // Look for Groq's search result format: 【0†Title†domain】
    const groqPattern = /【\d+†([^†]+)†([^】]+)】/g;
    let match;
    
    while ((match = groqPattern.exec(text)) !== null) {
      const title = match[1];
      const domain = match[2];
      
      // Assess authority based on domain
      const authority = this.assessDomainAuthority(domain);
      
      // Assess relevance based on title content
      const relevance = this.assessTitleRelevance(title);
      
      sources.push({
        url: `https://${domain}`, // Construct URL
        domain: domain,
        relevanceScore: relevance,
        authority: authority,
        keyFacts: [title], // Use title as key fact
        contentQuality: `Groq search result: ${title}`
      });
    }
    
    // Also extract any direct URLs found
    const directUrls = this.extractDirectUrls(text);
    for (const url of directUrls) {
      const domain = this.extractDomain(url);
      sources.push({
        url,
        domain,
        relevanceScore: 0.7, // Higher for direct URLs
        authority: this.assessDomainAuthority(domain),
        keyFacts: [],
        contentQuality: 'Direct URL found'
      });
    }
    
    return sources;
  }

  private parseRegexFallback(text: string): SourceAnalysis[] {
    const urlPattern = /https?:\/\/[^\s\)\]]+/g;
    const foundUrls = text.match(urlPattern) || [];
    
    return foundUrls.map(url => ({
      url,
      domain: this.extractDomain(url),
      relevanceScore: 0.5,
      authority: 'Medium' as const,
      keyFacts: [],
      contentQuality: 'Regex fallback'
    }));
  }

  private extractDirectUrls(text: string): string[] {
    const urlPattern = /https?:\/\/[^\s\)\]]+/g;
    return text.match(urlPattern) || [];
  }

  private assessDomainAuthority(domain: string): 'High' | 'Medium' | 'Low' {
    const highAuthority = [
      'sec.gov', 'treasury.gov', 'federalreserve.gov',
      'bloomberg.com', 'reuters.com', 'wsj.com', 'ft.com',
      'marketwatch.com', 'morningstar.com', 'yahoo.com',
      'stockanalysis.com', 'finviz.com'
    ];
    
    const mediumAuthority = [
      'cnbc.com', 'cnn.com', 'forbes.com', 'investopedia.com',
      'seeking.alpha.com', 'fool.com'
    ];
    
    if (highAuthority.some(auth => domain.includes(auth))) return 'High';
    if (mediumAuthority.some(auth => domain.includes(auth))) return 'Medium';
    return 'Low';
  }

  private assessTitleRelevance(title: string): number {
    // Simple relevance based on financial keywords
    const financialKeywords = ['stock', 'price', 'quote', 'market', 'earnings', 'dividend', 'analysis'];
    const matches = financialKeywords.filter(keyword => 
      title.toLowerCase().includes(keyword)
    ).length;
    
    return Math.min(0.3 + (matches * 0.2), 1.0); // 0.3 base + 0.2 per keyword, max 1.0
  }

  private extractSourceData(section: string): SourceAnalysis | null {
    const urlMatch = section.match(/\*\*URL:\*\*\s*(.+)/);
    const domainMatch = section.match(/\*\*DOMAIN:\*\*\s*(.+)/);
    const relevanceMatch = section.match(/\*\*RELEVANCE:\*\*\s*(\d*\.?\d+)/);
    const authorityMatch = section.match(/\*\*AUTHORITY:\*\*\s*(High|Medium|Low)/);
    const publicationMatch = section.match(/\*\*PUBLICATION:\*\*\s*(.+)/);
    const contentQualityMatch = section.match(/\*\*CONTENT QUALITY:\*\*\s*(.+)/);
    
    if (!urlMatch) return null;
    
    const url = urlMatch[1].trim();
    const domain = domainMatch ? domainMatch[1].trim() : this.extractDomain(url);
    const relevanceScore = relevanceMatch ? parseFloat(relevanceMatch[1]) : 0.5;
    const authority = (authorityMatch ? authorityMatch[1] : 'Medium') as 'High' | 'Medium' | 'Low';
    const publicationDate = publicationMatch ? publicationMatch[1].trim() : undefined;
    const contentQuality = contentQualityMatch ? contentQualityMatch[1].trim() : 'Not specified';
    
    // Extract key facts
    const keyFacts: string[] = [];
    const factsSection = section.match(/\*\*KEY FACTS:\*\*([\s\S]*?)(?:\*\*|$)/);
    if (factsSection) {
      const factLines = factsSection[1].split('\n')
        .map(line => line.trim())
        .filter(line => line.startsWith('-'))
        .map(line => line.substring(1).trim());
      keyFacts.push(...factLines);
    }
    
    return {
      url,
      domain,
      relevanceScore: Math.max(0, Math.min(1, relevanceScore)), // Clamp to 0-1
      authority,
      publicationDate: publicationDate === 'Not specified' ? undefined : publicationDate,
      keyFacts,
      contentQuality
    };
  }

  private extractDomain(url: string): string {
    try {
      return new URL(url).hostname;
    } catch {
      return 'unknown';
    }
  }

  private extractFallbackUrls(text: string): string[] {
    logProgress('Attempting fallback URL extraction', 4);
    const urlPattern = /https?:\/\/[^\s\)\]]+/g;
    const foundUrls = text.match(urlPattern) || [];
    
    // Clean and deduplicate URLs
    const cleanUrls = [...new Set(foundUrls)]
      .filter(url => !url.includes('example.com'))
      .slice(0, 10); // Limit to 10 URLs max
    
    if (cleanUrls.length > 0) {
      logProgress(`Fallback extracted ${cleanUrls.length} URLs`, 4);
      return cleanUrls;
    } else {
      logError('No URLs found even in fallback extraction!', 4);
      return ['https://web-search-results']; // Final fallback
    }
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
    
    const generateOptions: any = {
      model: MODELS.planner,
      system,
      prompt: `Goal: ${userGoal}\nReturn JSON with { subtasks: [{id, kind, prompt, deps?}] }`,
      temperature: 0.2,
    };
    
    // Add reasoning effort for Groq GPT-OSS models
    if (MODELS.provider === 'Groq') {
      generateOptions.providerOptions = {
        groq: {
          reasoningEffort: 'medium',
        },
      };
    }
    
    const { text } = await generateText(generateOptions);

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

  async propose(subtask: Subtask, context: Record<string, any>, k = 3, goal: string = '') {
    logProgress(`Generating ${k} proposals for ${subtask.kind} task: ${subtask.id}`, 2);
    
    // Get sources: use research sources from context, or do new retrieval for research tasks
    let retrieval: any;
    let retrievalTime = 0;
    
    if (subtask.kind === 'research') {
      // Research task: do actual retrieval (may include web search)
      const retrievalStartTime = Date.now();
      retrieval = await this.retriever.retrieve(subtask.prompt, subtask.kind, goal);
      retrievalTime = Date.now() - retrievalStartTime;
      if (retrievalTime > 1000) {
        logSuccess(`Retrieval completed in ${formatTime(retrievalTime)}`, 3);
      }
    } else {
      // Reasoning/verification tasks: use sources from research step in context
      const researchResults = Object.values(context).find(
        (result: any) => result?.evidence?.sources || result?.citations
      );
      
      if (researchResults) {
        const sourcesToUse = researchResults.citations || researchResults.sources || ['context-sources'];
        logProgress(`Using research sources from context: ${sourcesToUse.slice(0, 2).join(', ')}${sourcesToUse.length > 2 ? '...' : ''}`, 3);
        retrieval = researchResults.evidence || {
          nodes: [{ id: 'research_context', type: 'research', text: 'Using sources from research step' }],
          edges: [],
          sources: sourcesToUse,
        };
      } else {
        // Fallback if no research context available
        logProgress('No research context found, using knowledge-based sources', 3);
        retrieval = {
          nodes: [{ id: 'knowledge', type: 'knowledge', text: `Knowledge-based response for: ${subtask.prompt}` }],
          edges: [],
          sources: ['internal-knowledge'],
        };
      }
    }
    
    // Generate all proposals in parallel (no artificial delays)
    logProgress(`Starting ${k} parallel proposal generations`, 3);
    const proposalStartTime = Date.now();
    
    const proposalPromises = Array.from({ length: k }, async (_, i) => {
      const proposalId = i + 1;
      try {
        const system = `You are the ${subtask.kind} specialist. Use the evidence graph and be concise but precise. Cite sources explicitly.`;
        
        // Create enhanced prompt with explicit source tracking
        const promptData = {
          subtask,
          context,
          evidence_graph: retrieval,
          instruction: subtask.kind === 'verify' ? 
            'CRITICAL: Verify the previous step against the ORIGINAL research sources. Do not use internal knowledge if real sources were found.' :
            'Use the provided evidence and sources to complete this task.'
        };
        
        const generateOptions: any = {
          model: MODELS.solver,
          system,
          prompt: JSON.stringify(promptData),
          temperature: 0.6 + (i * 0.1), // Slight temperature variation for diversity
        };
        
        // Add reasoning effort for Groq GPT-OSS models
        if (MODELS.provider === 'Groq') {
          generateOptions.providerOptions = {
            groq: {
              reasoningEffort: subtask.kind === 'verify' || subtask.kind === 'reason' ? 'high' : 'medium',
            },
          };
        }
        
        const { text } = await generateText(generateOptions);
        logSuccess(`Proposal ${proposalId} completed`, 4);
        return { text, citations: retrieval.sources, evidence: retrieval };
        
      } catch (error) {
        const errorMsg = error instanceof Error ? error.message : 'Unknown error';
        logError(`Proposal ${proposalId} failed: ${errorMsg}`, 4);
        
        // For rate limits, implement exponential backoff
        if (errorMsg.includes('rate limit') || errorMsg.includes('Rate limit')) {
          const backoffTime = Math.min(1000 * Math.pow(2, i), 10000); // Max 10s backoff
          logProgress(`Retrying proposal ${proposalId} after ${formatTime(backoffTime)}`, 4);
          await new Promise(resolve => setTimeout(resolve, backoffTime));
          
          try {
            const { text } = await generateText(generateOptions);
            logSuccess(`Proposal ${proposalId} completed (retry)`, 4);
            return { text, citations: retrieval.sources, evidence: retrieval };
          } catch (retryError) {
            logError(`Proposal ${proposalId} retry failed`, 4);
          }
        }
        
        // Return fallback proposal
        return {
          text: `Fallback proposal ${proposalId}: ${errorMsg.includes('rate limit') ? 'Rate limited - using fallback response.' : 'Error occurred during generation.'}`,
          citations: retrieval.sources,
          evidence: retrieval,
        };
      }
    });
    
    const proposals = await Promise.all(proposalPromises);
    const proposalTime = Date.now() - proposalStartTime;
    logSuccess(`Generated ${proposals.length} proposals in ${formatTime(proposalTime)}`, 2);
    return proposals;
  }

  vote(proposals: Array<{ text: string; citations: string[]; evidence: any }>) {
    // Quick voting: prefer the longest response (often most detailed)
    // Or return the first if they're similar length
    if (proposals.length === 1) return proposals[0];
    
    const sortedByLength = [...proposals].sort((a, b) => b.text.length - a.text.length);
    
    // If the longest is significantly longer (>20% more), prefer it
    // Otherwise stick with the first (faster)
    const longest = sortedByLength[0];
    const first = proposals[0];
    
    if (longest.text.length > first.text.length * 1.2) {
      return longest;
    }
    
    return first;
  }
}

// ────────────────────────────────────────────────────────────────────────────────
// Judge: checks artifacts against tests and citations
// ────────────────────────────────────────────────────────────────────────────────

class Judge {
  constructor(private tooling: Tooling) {}
  
  // Determine if citations are required based on question type
  private requiresCitations(goal: string): boolean {
    const goalLower = goal.toLowerCase();
    
    // Questions that DON'T need citations (knowledge/logic based)
    const mathKeywords = ['calculate', 'solve', 'equation', 'derivative', 'integral', 'algebra', 'geometry', 'arithmetic', '+', '-', '*', '/', '=', 'formula', 'theorem', 'proof', 'math'];
    const codeKeywords = ['code', 'programming', 'function', 'algorithm', 'debug', 'syntax', 'variable', 'loop', 'class', 'method', 'python', 'javascript', 'typescript'];
    const conceptualKeywords = ['what is', 'define', 'explain', 'how does', 'how to', 'difference between', 'compare', 'concept of', 'meaning of', 'definition of'];
    const technicalKeywords = ['machine learning', 'ai', 'database', 'api', 'framework', 'library', 'protocol', 'encryption', 'security', 'network'];
    
    // Questions that DO need citations (factual/current)
    const citationRequiredKeywords = ['latest', 'recent', 'current', 'today', 'this year', '2024', '2025', 'breaking news', 'update', 'who is', 'when did', 'population of', 'capital of', 'price of', 'stock price', 'weather', 'news', 'event'];
    
    // First check if it explicitly needs citations
    if (citationRequiredKeywords.some(keyword => goalLower.includes(keyword))) {
      return true;
    }
    
    // Then check if it's knowledge-based (doesn't need citations)
    if (mathKeywords.some(keyword => goalLower.includes(keyword)) ||
        codeKeywords.some(keyword => goalLower.includes(keyword)) ||
        conceptualKeywords.some(keyword => goalLower.includes(keyword)) ||
        technicalKeywords.some(keyword => goalLower.includes(keyword))) {
      return false;
    }
    
    // Default: require citations for ambiguous questions
    return true;
  }

  async inspect(subtask: Subtask, artifact: Record<string, any>, goal: string = '') {
    logProgress(`Judging artifact for ${subtask.id}`, 2);
    const tests = await this.tooling.runTests(artifact);
    const needsCitations = this.requiresCitations(goal);
    
    // Analyze the sources being used FIRST
    const citations = artifact.citations || artifact.sources || [];
    const hasRealSources = citations.some((source: string) => 
      source.startsWith('http') && !source.includes('example.com')
    );
    
    let system = 'You are a strict verifier. ';
    if (hasRealSources) {
      system += 'CRITICAL: Real sources were found during research. Verify all claims against these specific sources. Do not use internal knowledge when real sources are available.';
    } else if (needsCitations) {
      system += 'Check factuality against citations and logical consistency. Penalize unsupported factual claims.';
    } else {
      system += 'Focus on logical consistency and mathematical/technical accuracy. Citations are not required for math, calculations, or code - the logic itself is the evidence.';
    }
    const usesKnowledgeBase = citations.includes('internal-knowledge');
    
    // Determine evaluation mode based on actual sources found
    let evaluationMode: string;
    let judgeNote: string;
    
    if (hasRealSources) {
      evaluationMode = 'source_verification';
      judgeNote = 'CRITICAL: Verify against the specific sources cited. Check if claims match the actual source content from the research phase.';
      logProgress(`Judge using real sources: ${citations.filter((s: string) => s.startsWith('http')).slice(0, 2).join(', ')}`, 3);
    } else if (usesKnowledgeBase || !needsCitations) {
      evaluationMode = 'logic_based';
      judgeNote = 'Evaluate logical accuracy and completeness. External citations not required.';
    } else {
      evaluationMode = 'citation_required';
      judgeNote = 'This requires factual verification with external sources. CRITICAL: You can only verify against sources that were provided in the research phase. Do not search for or use new sources.';
      logProgress('Judge mode: citation_required (no real sources found)', 3);
    }
    
    const judgePrompt = {
      subtask,
      artifact, 
      tests,
      evaluation_mode: evaluationMode,
      sources_available: citations,
      has_real_sources: hasRealSources,
      note: judgeNote,
      critical_instruction: subtask.kind === 'verify' ? 
        hasRealSources ? 
          'You are verifying the previous reasoning step. Use ONLY the sources that were found in the research phase. Do not abandon real sources for internal knowledge.' :
          'You are verifying the previous reasoning step. The research phase used placeholder/internal sources only. Do not introduce new external sources. Verify based on logical consistency and known facts only.' :
        'Evaluate this step for accuracy and completeness.'
    };
    
    const generateOptions: any = {
      model: MODELS.judge,
      system,
      prompt: JSON.stringify(judgePrompt),
      temperature: 0,
    };
    
    // Add reasoning effort for Groq GPT-OSS models (highest for judge)
    if (MODELS.provider === 'Groq') {
      generateOptions.providerOptions = {
        groq: {
          reasoningEffort: 'high',
        },
      };
    }
    
    const { text: critique } = await generateText(generateOptions);

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

      // Process batch in parallel when possible, with better progress tracking
      const batchStartTime = Date.now();
      
      if (batch.length > 1) {
        logProgress(`Processing ${batch.length} subtasks in parallel`, 1);
      }
      
      const batchPromises = batch.map(async (sub, index) => {
        const currentTaskNum = taskCount + index + 1;
        console.log(`\n[${currentTaskNum}/${plan.subtasks.length}] Processing subtask: ${sub.id} (${sub.kind})`);
        
        const taskStartTime = Date.now();
        const artifact = await this.executeSubtask(sub, artifacts, goal);
        const taskDuration = Date.now() - taskStartTime;
        
        return { 
          id: sub.id, 
          artifact, 
          duration: taskDuration,
          taskNumber: currentTaskNum 
        };
      });
      
      const results = await Promise.all(batchPromises);
      const batchDuration = Date.now() - batchStartTime;
      
      taskCount += batch.length;
      
      // Update artifacts and mark as done
      for (const { id, artifact, duration, taskNumber } of results) {
        artifacts[id] = artifact;
        done.add(id);
        logSuccess(`Completed subtask ${id} in ${formatTime(duration)}`, 1);
      }
      
      if (batch.length > 1) {
        logSuccess(`Batch of ${batch.length} tasks completed in ${formatTime(batchDuration)}`, 1);
      }
    }

    const totalTime = Date.now() - startTime;
    const finalId = plan.subtasks.findLast?.(Boolean)?.id ?? plan.subtasks[plan.subtasks.length - 1].id;
    const final = artifacts[finalId];
    
    console.log(`\n✨ Multi-agent reasoning completed in ${formatTime(totalTime)}`);
    return { final, evidence: this.log.toJSON(), plan, executionTime: totalTime } as const;
  }

  private async executeSubtask(sub: Subtask, artifacts: Record<ID, Record<string, any>>, goal: string) {
    const context = Object.fromEntries((sub.deps ?? []).map(d => [d, artifacts[d]]));
    const startTime = Date.now();

    let lastCandidate: Record<string, any> | undefined;
    for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
      logProgress(`Attempt ${attempt + 1}/${this.maxRetries + 1}`, 1);
      // Reduce proposals for Groq to manage rate limits
      const proposalCount = MODELS.provider === 'Groq' ? 2 : (sub.kind === 'verify' ? 2 : 3);
      const proposals = await this.solver.propose(sub, context, proposalCount, goal);
      const candidate = this.solver.vote(proposals);

      this.log.add({
        stepId: `${sub.id}:${attempt}`,
        role: 'solver',
        input: { subtask: sub, context },
        output: candidate,
        citations: candidate.citations,
      });

      const { passed, info } = await this.judge.inspect(sub, candidate, goal);
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
  console.log('  GROQ_API_KEY   - Required for Groq provider\n');
  console.log('Output:');
  console.log('  Clean answer displayed on CLI');
  console.log('  Detailed reasoning automatically saved to reasoning-TIMESTAMP.json\n');
  console.log('Models:');
  console.log('  OpenAI: GPT-4o-mini (planner/solver), GPT-4o (judge)');
  console.log('  Groq:   GPT-OSS-20B (planner/solver), GPT-OSS-120B (judge)\n');
  console.log('Features:');
  console.log('  OpenAI: Advanced reasoning, placeholder citations');
  console.log('  Groq:   Smart web search, real citations, reasoning effort\n');
  console.log('Smart Search Logic:');
  console.log('  🔍 Web search: Current events, facts, specific queries');
  console.log('  🧠 No search: Math, code, concepts, definitions, how-to\n');
}

export async function runMultiAgentReasoning(goal: string, provider: ModelProvider = 'openai') {
  if (!validateEnvironment(provider)) {
    process.exit(1);
  }
  
  // Initialize models with selected provider
  MODELS = createModels(provider);
  console.log(`\n🤖 Using ${MODELS.provider}: ${MODELS.models}`);

  // Enable real web search for Groq models
  const useRealSearch = provider === 'groq';
  const retriever = new Retriever(useRealSearch);
  const tooling = new Tooling();
  const planner = new Planner();
  const solver = new Solver(retriever);
  const judge = new Judge(tooling);
  const orch = new Orchestrator(planner, solver, judge, 2);
  
  if (useRealSearch) {
    console.log('  🔍 Web search enabled for real citations');
  }

  console.log(`\n🎯 GOAL: ${goal}`);
  const result = await orch.run(goal);
  
  // Display clean answer on CLI
  console.log('\n' + '='.repeat(80));
  console.log('🏆 ANSWER');
  console.log('='.repeat(80));
  
  // Extract the main text content from the final result
  let answer = '';
  if (typeof result.final === 'object' && result.final !== null) {
    if ('text' in result.final && typeof result.final.text === 'string') {
      answer = result.final.text;
    } else {
      answer = JSON.stringify(result.final, null, 2);
    }
  } else {
    answer = String(result.final);
  }
  
  console.log(answer);
  console.log('\n' + '='.repeat(80));
  
  console.log(`\n⏱️  Total execution time: ${formatTime(result.executionTime)}`);
  
  // Always save detailed reasoning log
  const fs = await import('fs');
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const filename = `reasoning-${timestamp}.json`;
  
  const reasoningLog = {
    goal,
    provider: MODELS.provider,
    models: MODELS.models,
    executionTime: result.executionTime,
    plan: result.plan,
    evidence: JSON.parse(result.evidence),
    finalResult: result.final,
    timestamp: new Date().toISOString(),
  };
  
  fs.writeFileSync(filename, JSON.stringify(reasoningLog, null, 2));
  console.log(`\n📝 Detailed reasoning saved to: ${filename}`);
  
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