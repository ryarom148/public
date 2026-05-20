# Claude Code — Windows Setup: FastCode + CodeGraph + Skills
## Instructions for Claude Code to execute
 
> **How to use this file:**
> Open PowerShell, navigate to your project, start Claude Code with `claude`, then say:
> `"Read SETUP_CLAUDE_CODE_WINDOWS.md and execute every phase in order. Stop and report to me if any step fails."`
 
---
 
## Mental Model Before Starting
 
```
FastCode  = WHAT does this code mean? (semantic Q&A, cross-file understanding, costs tokens)
CodeGraph = WHERE is everything and HOW is it connected? (AST graph, zero tokens, always first)
Claude    = Write it. Explain it. Document it. Test it.
```
 
### Tool decision rule — apply before every query
 
```
Can CodeGraph answer this with a structural query?
  YES → Use CodeGraph. It's free, instant, and local.
  NO  → Is FastCode connected? (/mcp to check)
          YES → Use code_qa with the specific question and repo path.
          NO  → Read only the minimal files CodeGraph identified. Use grep as last resort.
```
 
---
 
## FastCode MCP Tools — Complete Reference
Source: https://github.com/HKUDS/FastCode (README, MCP Server section)
 
FastCode exposes **6 MCP tools** through `mcp_server.py`.
 
### Tool 1: `code_qa`
**The primary tool.** Ask any semantic question about one or more repositories.
 
| Parameter | Type | Required | Description |
|---|---|---|---|
| `repos` | list[str] | yes | Local paths (`C:/myproject`) or GitHub URLs. Multiple repos supported. Auto-clones URLs to `./repos/`. |
| `query` | str | yes | Natural language question about the code. |
| `session_id` | str | no | Reuse a previous session to continue a multi-turn conversation. Returned in each response — pass it back for follow-ups. |
| `multi_turn` | bool | no | Default `true`. Uses prior Q&A from the same `session_id` for query rewriting and answer generation. |
 
**When to use `code_qa`:**
- "How does feature X work end-to-end?" → CodeGraph cannot trace semantic meaning, only structure
- "What is the architectural relationship between module A and B?" → requires understanding, not just edges
- "Where should I add rate limiting?" → requires judgment across files
- "Explain what this subsystem does" → cross-file comprehension
- "Which files would be affected if I change the User model?" → follow-up in same session
- Any question CodeGraph returns no result for, or where the structural answer is incomplete
**Multi-repo example:**
```
Use code_qa with repos=["/path/to/frontend", "/path/to/backend"] to explain
how the authentication token flows from login to API authorization.
```
 
**Multi-turn example (continue a session):**
```
# First call — starts a new session, FastCode returns a session_id
Use code_qa on /path/to/repo: "Explain the payment processing flow"
 
# Second call — pass the session_id back to continue
Use code_qa on /path/to/repo with session_id=[returned id]: "Which files would I need to change to add PayPal support?"
```
 
**Auto-clone example (index a remote repo without cloning manually):**
```
Use code_qa with repos=["https://github.com/org/repo"] to explain the architecture.
FastCode will clone and index it automatically.
```
 
---
 
### Tool 2: `list_repos`
List all repositories that have been indexed by FastCode and are available for querying.
 
**When to use:**
- At the start of a session to see what is already indexed
- Before calling `code_qa` to confirm the repo path is correct
- When unsure whether a repo needs re-indexing
**Example:**
```
Use list_repos to show all FastCode-indexed repositories before I ask questions about the codebase.
```
 
---
 
### Tool 3: `list_sessions`
List all existing conversation sessions with their titles and turn counts.
 
**When to use:**
- To find a previous session on a topic you already explored
- Before starting a new session, to check if relevant context already exists
**Example:**
```
Use list_sessions to show prior conversations about this repo so I can continue one instead of starting fresh.
```
 
---
 
### Tool 4: `get_session`
Retrieve the full Q&A history of a specific session.
 
**When to use:**
- To review what was already analyzed before continuing work
- To share prior analysis with a team member
- When resuming work on a feature after a break
**Example:**
```
Use get_session with session_id=[id] to show the full conversation history about the auth module.
```
 
---
 
### Tool 5: `delete_session`
Delete a conversation session and all its history.
 
**When to use:**
- To clean up stale sessions after a refactor
- To reset context when analysis is outdated
---
 
### Tool 6: `remove_repo`
Delete indexed metadata for a repository (`.faiss`, `_metadata.pkl`, `_bm25.pkl`, `_graphs.pkl`) and remove its entry from `repo_overviews.pkl`. Keeps the source code — only deletes the index.
 
**When to use:**
- After a major refactor that makes the old index stale
- To force a full re-index on the next `code_qa` call
- To free disk space from old indexed repos
**Example:**
```
Use remove_repo for /path/to/repo, then call code_qa again to trigger a fresh full index.
```
 
---
 
## CodeGraph MCP Tools — Complete Reference
Source: https://github.com/colbymchenry/codegraph (README, Medium article by author)
 
CodeGraph exposes **6 MCP tools** backed by a local SQLite graph (tree-sitter AST, zero API cost).
 
### Tool 1: `codegraph_explore`
**The primary tool.** One call returns entry points, related symbols, and code snippets. The recommended first call for any new question about the codebase.
 
**When to use:**
- First question in any exploration — "How does X work?"
- Before reading any files
- When CodeGraph is available and the question is structural
**Example:**
```
Use codegraph_explore to answer: "How does authentication work in this codebase?"
```
 
---
 
### Tool 2: `codegraph_search`
Search symbols by name. Returns **locations only** (file + line) — not code content. Fast, zero-token lookup.
 
**When to use:**
- "Where is class UserService defined?"
- "Which files contain a function called `processPayment`?"
- Finding the entry point before reading it
**Example:**
```
Use codegraph_search to find all symbols named "login" or "authenticate".
```
 
---
 
### Tool 3: `codegraph_context`
Build comprehensive context for a specific task. Returns the relevant symbols, their relationships, and code snippets tuned to the task description.
 
**When to use:**
- Before implementing a feature — get all relevant context in one call
- Before writing documentation for a module
- When you need more than just locations
**Example:**
```
Use codegraph_context with task="Add retry logic to the payment processor" to get
all relevant files, functions, and relationships before I write any code.
```
 
---
 
### Tool 4: `codegraph_callers`
Find everything that calls a given function or method. Returns caller locations and the call chain.
 
**When to use:**
- "What uses this function?" before changing its signature
- Understanding the blast radius before a refactor
- Tracing execution flow backward from a bug
**Example:**
```
Use codegraph_callers on function "processOrder" to see everything that calls it
before I change its return type.
```
 
---
 
### Tool 5: `codegraph_callees`
Find everything a given function calls — its internal dependencies.
 
**When to use:**
- Understanding what a function depends on before testing it in isolation
- Tracing execution flow forward from an entry point
- Finding all side effects of calling a function
**Example:**
```
Use codegraph_callees on "checkout" to see the full chain of function calls it triggers.
```
 
---
 
### Tool 6: `codegraph_impact`
Calculate the full change blast radius for a symbol. Shows direct callers, indirect callers, and the complete dependency chain.
 
**When to use — always before refactoring:**
- Before changing a function signature
- Before renaming a class
- Before deleting a module
- Before changing a database model
**Example:**
```
Use codegraph_impact on class "UserSession" to show everything that would break
if I change its constructor signature.
```
 
---
 
## When CodeGraph Cannot Answer — Escalation to FastCode
 
CodeGraph is structural and deterministic. It answers questions about **what exists and how it is connected**. It cannot answer questions that require **understanding meaning**.
 
### CodeGraph cannot answer — use FastCode `code_qa` instead:
 
| Question type | Why CodeGraph can't answer | FastCode approach |
|---|---|---|
| "How does X feature work conceptually?" | CodeGraph shows the graph, not the meaning | `code_qa` with `query="Explain how X works"` |
| "Is this implementation correct?" | Requires reasoning, not graph traversal | `code_qa` with `query="Review the implementation of X for correctness"` |
| "What is the purpose of this module?" | Semantic intent, not structure | `code_qa` with `query="What does module X do and why does it exist?"` |
| "Where should I add Y to follow existing patterns?" | Requires judgment, not lookup | `code_qa` with `query="Where should I add Y to follow the existing pattern?"` |
| "What breaks if I change X?" (complex/indirect) | `codegraph_impact` covers direct deps; FastCode reasons about indirect effects | `code_qa` with multi-turn session for follow-ups |
| "Summarize the architecture for a new developer" | Requires synthesis across the whole graph | `code_qa` with `query="Summarize the architecture for a new developer"` |
| Symbol not found in CodeGraph index | New file or unsupported language | `code_qa` with the file path directly |
| Cross-repo question | CodeGraph is per-project | `code_qa` with `repos=[path1, path2]` |
 
### Escalation protocol when CodeGraph returns no result:
```
1. Retry codegraph_search with a shorter/different symbol name (typos, partial names)
2. Try codegraph_explore with a broader question
3. If still no result: escalate to FastCode code_qa
4. If FastCode is unavailable: read only the file(s) most likely to contain the answer
   based on directory structure — do NOT grep the whole repo
```
 
---
 
## Phase 1 — Verify Prerequisites
 
```powershell
claude --version
git --version
node --version
npm --version
python --version
winget --version
```
 
**If Claude Code is missing:**
```powershell
irm https://claude.ai/install.ps1 | iex
# OR via winget:
winget install --id Anthropic.ClaudeCode -e --source winget
```
Close and reopen PowerShell. Verify with `claude --version`.
If still not found, add to PATH:
```powershell
[System.Environment]::SetEnvironmentVariable(
  "PATH",
  "$env:USERPROFILE\.local\bin;" + [System.Environment]::GetEnvironmentVariable("PATH","User"),
  "User"
)
```
 
**If Git is missing:** `winget install --id Git.Git -e --source winget`
**If Node.js is missing:** `winget install --id OpenJS.NodeJS.LTS -e --source winget`
**If Python is missing:** `winget install --id Python.Python.3.12 -e --source winget`
 
Close and reopen PowerShell after every winget install.
 
---
 
## Phase 2 — Install CodeGraph
 
Source: https://github.com/colbymchenry/codegraph
 
### Step 2a — Install and run interactive installer
 
```powershell
npm install -g @colbymchenry/codegraph
npx @colbymchenry/codegraph
```
 
Prompts: select **Claude Code**, then **global**. The installer writes the MCP server config and auto-allow permissions into `~/.claude.json`.
 
Non-interactive (for scripting):
```powershell
codegraph install --target=claude --location=global --yes
```
 
### Step 2b — Initialize and index your project
 
```powershell
cd "C:\PATH\TO\MY\PROJECT"
codegraph init
codegraph index
```
 
### Step 2c — Verify backend
 
```powershell
codegraph status
```
 
Must show `Backend: native`. If it shows `Backend: wasm` (5–10x slower):
```powershell
winget install --id Microsoft.VisualStudio.2022.BuildTools -e --source winget
npm rebuild better-sqlite3
codegraph status   # must now show Backend: native
```
 
### Step 2d — Verify inside Claude Code
 
```powershell
claude
```
```
/mcp
```
Confirm `codegraph` is listed. Test:
```
Use codegraph_explore to summarize the main modules and entry points.
Do not read any source files directly.
```
 
---
 
## Phase 3 — Install FastCode
 
Source: https://github.com/HKUDS/FastCode
 
FastCode is optional. If it cannot be installed, all skills below degrade gracefully to CodeGraph-only mode.
 
### Step 3a — Install uv
 
```powershell
pip install uv
uv --version
```
 
### Step 3b — Clone FastCode
 
```powershell
New-Item -ItemType Directory -Force -Path "C:\tools"
cd C:\tools
git clone https://github.com/HKUDS/FastCode.git
cd FastCode
```
 
### Step 3c — Read the README before installing
 
```powershell
type README.md
```
If the README shows a different install method, follow it. Steps below match v1.0.1.
 
### Step 3d — Create venv and install
 
```powershell
uv venv --python=3.12
.\.venv\Scripts\Activate.ps1       # Windows path — NOT source .venv/bin/activate
uv pip install -r requirements.txt
python -c "import fastcode; print('FastCode OK')"
deactivate
```
 
### Step 3e — Configure API key
 
```powershell
copy env.example .env
notepad .env
```
 
```env
# Works with any OpenAI-compatible provider
OPENAI_API_KEY=sk-your-key-here
MODEL=gpt-4o
BASE_URL=https://api.openai.com/v1
 
# For Anthropic Claude API:
# OPENAI_API_KEY=sk-ant-your-key-here
# MODEL=claude-sonnet-4-6
# BASE_URL=https://api.anthropic.com/v1
```
 
### Step 3f — Test server manually
 
```powershell
cd C:\tools\FastCode
.\.venv\Scripts\Activate.ps1
python mcp_server.py
# Should start without errors. Ctrl+C to stop.
deactivate
```
 
### Step 3g — Register with Claude Code
 
```powershell
claude mcp add fastcode -- "C:\tools\FastCode\.venv\Scripts\python.exe" "C:\tools\FastCode\mcp_server.py"
```
 
Or edit `%USERPROFILE%\.claude\claude_desktop_config.json` directly:
```json
{
  "mcpServers": {
    "fastcode": {
      "command": "C:\\tools\\FastCode\\.venv\\Scripts\\python.exe",
      "args": ["C:\\tools\\FastCode\\mcp_server.py"],
      "env": {
        "MODEL": "gpt-4o",
        "BASE_URL": "https://api.openai.com/v1",
        "OPENAI_API_KEY": "sk-your-key-here"
      }
    }
  }
}
```
 
### Step 3h — Verify
 
```
/mcp
```
Both `codegraph` and `fastcode` must be listed.
 
```
Use list_repos to show all FastCode-indexed repositories.
```
Empty list is correct — repos are indexed on first `code_qa` call.
 
---
 
## Phase 4 — Create Project Structure
 
```powershell
New-Item -ItemType Directory -Force -Path ".claude"
New-Item -ItemType Directory -Force -Path ".claude\skills\repo-navigation"
New-Item -ItemType Directory -Force -Path ".claude\skills\documentation-writer"
New-Item -ItemType Directory -Force -Path ".claude\skills\tutorial-writer"
New-Item -ItemType Directory -Force -Path ".claude\skills\security-review"
New-Item -ItemType Directory -Force -Path ".claude\skills\feature-planner"
New-Item -ItemType Directory -Force -Path "docs\TUTORIALS"
```
 
> **Skill locations:**
> `.claude/skills/` = project-scoped, committed to git, shared with team
> `~/.claude/skills/` = personal, available across all projects
 
---
 
## Phase 5 — CLAUDE.md
 
```powershell
notepad CLAUDE.md
```
 
```markdown
# Project Operating Instructions
 
## Tool Responsibilities
 
**CodeGraph** — structural truth layer (local, zero tokens, always available)
Source: https://github.com/colbymchenry/codegraph
MCP tools: codegraph_explore, codegraph_search, codegraph_context, codegraph_callers, codegraph_callees, codegraph_impact
Use for: symbol location, call graphs, import graphs, dependency chains, blast-radius analysis, entry-point discovery.
Always try CodeGraph first. It costs nothing.
 
**FastCode** — semantic understanding layer (requires LLM API, costs tokens, use selectively)
Source: https://github.com/HKUDS/FastCode
MCP tools: code_qa (primary), list_repos, list_sessions, get_session, delete_session, remove_repo
Use for: conceptual explanations, cross-file semantic reasoning, architecture summaries, questions CodeGraph cannot answer structurally.
Check `/mcp` before using FastCode. If unavailable, CodeGraph + targeted file reads cover most tasks.
 
**Claude Code** — reasoning and output layer
Use for: writing code, writing tests, writing documentation, explaining behavior, security analysis, feature planning.
 
## Exploration Rule — apply before every query
 
```
Can CodeGraph answer this structurally?
  YES → Use CodeGraph. Free, instant, local.
  NO  → Is FastCode connected?
          YES → Use code_qa with the repo path and specific question.
                Reuse session_id for follow-up questions in the same session.
          NO  → Read only minimal files identified by CodeGraph structure.
                Use grep only as a last resort for a specific string.
```
 
Never start a session with broad grep across the whole repository.
 
## FastCode Escalation Triggers
Use code_qa when:
- CodeGraph returns no result after retry with different symbol name
- The question requires semantic understanding, not structural lookup
- The question spans multiple repos (pass both paths in repos=[])
- A new developer needs an architecture overview
- You need to continue prior analysis (reuse session_id from list_sessions)
 
## FastCode Session Management
- Before a long analysis: call list_repos to confirm the repo is indexed
- Before starting a new session: call list_sessions to find existing relevant sessions
- For follow-up questions: always pass the session_id back to code_qa
- After a major refactor: call remove_repo then code_qa to force fresh indexing
 
## Documentation Rule
1. Use repo-navigation skill first.
2. Verify exact file/function relationships with CodeGraph before writing anything.
3. Do not describe architecture not present in the code.
4. Save docs to `/docs/`. Save tutorials to `/docs/TUTORIALS/`.
5. Include exact source file and function references in every technical document.
 
## Security Rule
1. Use CodeGraph for source-to-sink structural tracing (codegraph_callers, codegraph_callees).
2. Use FastCode code_qa for semantic reasoning about whether a path is exploitable.
3. Report only findings backed by a file, function, and traced data path.
4. No vague findings.
```
 
---
 
## Phase 6 — Skills
 
### Skill 1: repo-navigation
 
```powershell
notepad .claude\skills\repo-navigation\SKILL.md
```
 
```markdown
---
name: repo-navigation
description: >
  Efficiently explore any repository to locate code, trace behavior, map dependencies,
  or prepare context before editing or documenting. Use when asked to understand
  the codebase, find where something is implemented, or plan a feature or doc task.
  Applies the CodeGraph-first, FastCode-second rule. Never starts with broad grep.
---
 
# Repo Navigation Skill
Source: https://github.com/colbymchenry/codegraph | https://github.com/HKUDS/FastCode
 
## Decision Rule (apply every time)
 
```
Can CodeGraph answer this with a structural query?
  YES → Use CodeGraph. Zero tokens. Always first.
  NO, or CodeGraph returns nothing after retry → Use FastCode code_qa.
  FastCode unavailable → Read only files CodeGraph identified. Grep as last resort.
```
 
## Step 1 — CodeGraph structural exploration (always first)
 
Use these tools in order, stopping when you have enough to act:
 
| Tool | Question it answers | When to call it |
|---|---|---|
| `codegraph_explore` | What are the main entry points, modules, and relationships? | First call for any new topic |
| `codegraph_search` | Where is symbol X defined? (returns file + line, no code) | When you know a name |
| `codegraph_context` | What context is needed for task T? | Before implementing or documenting |
| `codegraph_callers` | What calls function X? | Before changing a function signature |
| `codegraph_callees` | What does function X call? | Understanding internal dependencies |
| `codegraph_impact` | What breaks if I change symbol X? | Before every refactor |
 
**CodeGraph cannot answer — escalate to Step 2:**
- Semantic/conceptual questions ("how does X work in terms of product behavior")
- Cross-repo questions (two separate codebases)
- Symbol not found after retry with a shorter name
- Questions requiring judgment across many files ("where should I add Y?")
 
## Step 2 — FastCode semantic scouting (if available, after CodeGraph)
 
Check `/mcp` first. If `fastcode` is listed:
 
**Primary tool: `code_qa`**
```
Parameters:
  repos: ["/absolute/path/to/project"]   ← required, local path or GitHub URL
  query: "Your specific question"         ← required, natural language
  session_id: "prior-session-id"         ← optional, for multi-turn follow-ups
  multi_turn: true                        ← default true, enables context reuse
```
 
**Session management before querying:**
1. Call `list_repos` — confirm repo is indexed (first call will auto-index)
2. Call `list_sessions` — find prior sessions on this topic to reuse
3. If resuming: call `get_session` with the session_id to review prior analysis
4. Call `code_qa` — pass session_id if continuing, omit for a new session
5. Save the returned session_id for follow-up questions
 
**When to use `code_qa` specifically:**
- "Explain the end-to-end flow of [feature]"
- "What is the architectural purpose of [module]?"
- "Where is the best place to add [behavior] to match existing patterns?"
- "What would change if I refactor [class]?" (deeper than codegraph_impact)
- "Summarize this codebase for a new developer"
- Any CodeGraph result that needs conceptual interpretation
 
**Multi-repo queries:**
```
Use code_qa with repos=["/path/to/frontend", "/path/to/backend"]
to explain how the authentication token flows across both services.
```
 
**Re-indexing after major refactor:**
```
1. Call remove_repo with the repo path to clear the stale index
2. Call code_qa — FastCode will re-index automatically on the next query
```
 
## Step 3 — Targeted file reads (only after Steps 1–2)
Read only the exact files CodeGraph or FastCode identified. No speculative reads.
 
## Step 4 — Targeted grep (last resort only)
Use grep only when looking for a specific string not found by symbol search.
Never grep the whole repo as an exploration strategy.
 
## If Both Tools Unavailable
Tell the user: "CodeGraph and FastCode are both unavailable. Run `/mcp` to check status."
Ask for explicit permission before falling back to directory browsing.
 
## Required Output Format
1. CodeGraph findings — which tools were called, what was returned
2. FastCode findings — if called, what session_id was used, what was learned
3. Files identified as relevant (with source: CodeGraph / FastCode / directory inference)
4. Files explicitly excluded and why
5. Confidence level in the structural understanding
6. Recommended next action
```
 
---
 
### Skill 2: documentation-writer
 
```powershell
notepad .claude\skills\documentation-writer\SKILL.md
```
 
```markdown
---
name: documentation-writer
description: >
  Create or update accurate project documentation grounded in verified repository
  structure. Use when asked to write architecture docs, API references, workflow
  guides, getting-started docs, or any technical documentation. Always verifies
  with CodeGraph before writing. Uses FastCode for conceptual synthesis when needed.
---
 
# Documentation Writer Skill
Source: https://github.com/colbymchenry/codegraph | https://github.com/HKUDS/FastCode
 
## Required Workflow
 
### Step 1 — Structural verification with CodeGraph
Before writing a single sentence, run:
1. `codegraph_explore` — get entry points, main modules, key relationships
2. `codegraph_search` — find exact symbols that will be referenced in the doc
3. `codegraph_context` — build context for the specific doc being written
4. `codegraph_callers` / `codegraph_callees` — verify flows described in the doc
 
Do not document any module, function, or flow that CodeGraph cannot verify exists.
 
### Step 2 — Conceptual synthesis with FastCode (when needed)
Use `code_qa` when:
- The doc requires explaining *why* something works the way it does, not just *what* it is
- An architecture overview needs synthesis across many files
- The audience is a new developer who needs conceptual orientation, not just symbol lists
- CodeGraph confirms the structure but the doc needs narrative explanation
 
**FastCode usage for documentation:**
```
# Before writing ARCHITECTURE.md — get a synthesis
Use code_qa with:
  repos: ["/path/to/project"]
  query: "Explain the overall architecture for a new developer. What are the main
          layers, what does each do, and how do they interact?"
 
# Save the session_id. Use it for follow-up questions about specific sections:
Use code_qa with:
  repos: ["/path/to/project"]
  session_id: [saved id]
  query: "Now explain only the authentication subsystem in more detail."
```
 
### Step 3 — Write from verified evidence
- Every file path: verified by CodeGraph
- Every function/class name: exact names from CodeGraph search results
- Every flow description: verified by codegraph_callers / codegraph_callees
- Architecture narrative: grounded in FastCode code_qa when used
 
Mark anything unverified as `[UNVERIFIED — needs review]`.
 
## Default Output Paths
 
| Document | Path |
|---|---|
| Architecture overview | `docs/ARCHITECTURE.md` |
| Getting started | `docs/GETTING_STARTED.md` |
| Core workflows | `docs/CORE_WORKFLOWS.md` |
| API reference | `docs/API_REFERENCE.md` |
| Feature development guide | `docs/FEATURE_DEVELOPMENT_GUIDE.md` |
| Security model | `docs/SECURITY_MODEL.md` |
 
## Required Document Structure
 
Every technical document must include:
1. **Purpose** — what this doc covers and who it is for
2. **Relevant source files** — exact paths from CodeGraph
3. **Main flow** — entry to output, step by step, verified by callers/callees
4. **Key functions/classes** — exact names and what they do
5. **Dependencies** — what this module imports (from CodeGraph)
6. **Extension points** — where to add new behavior
7. **Testing notes** — test files that cover this area
8. **Mermaid diagram** — when the flow has more than 3 steps
 
## Rules
- Do not describe features not confirmed by CodeGraph or FastCode analysis.
- Do not read entire files speculatively — use CodeGraph to narrow first.
- Save the document before reporting completion.
```
 
---
 
### Skill 3: tutorial-writer
 
```powershell
notepad .claude\skills\tutorial-writer\SKILL.md
```
 
```markdown
---
name: tutorial-writer
description: >
  Generate developer tutorials grounded in actual codebase behavior. Use when asked
  to create a step-by-step tutorial, walkthrough, or how-to guide for a specific
  feature, flow, or pattern. Saves output to docs/TUTORIALS/.
---
 
# Tutorial Writer Skill
Source: https://github.com/colbymchenry/codegraph | https://github.com/HKUDS/FastCode
 
## Required Workflow
 
### Step 1 — Locate the feature with CodeGraph
1. `codegraph_search` — find the entry point by symbol name
2. `codegraph_explore` — get the full structural picture of the feature area
3. `codegraph_callers` / `codegraph_callees` — trace the complete call chain
4. `codegraph_context` — build focused context including related tests
 
### Step 2 — Synthesize the flow with FastCode
After CodeGraph gives the structure, use `code_qa` to understand the *why*:
 
```
Use code_qa with:
  repos: ["/path/to/project"]
  query: "Walk me through how [feature] works step by step for a developer
          who wants to write a tutorial about it. Focus on the flow from
          user action to final output."
```
 
Save the session_id. If the tutorial covers multiple sections:
```
Use code_qa with:
  repos: ["/path/to/project"]
  session_id: [saved id]
  query: "Now focus on the error handling and edge cases for [feature]."
```
 
If FastCode is unavailable: use only CodeGraph callers/callees to trace the flow,
and read only the files CodeGraph identified.
 
### Step 3 — Write the tutorial
Use exact file names and function/class names from CodeGraph results.
Base narrative on FastCode code_qa analysis where available.
 
## Required Tutorial Structure
 
```
# Tutorial: [Title]
 
## What You Will Learn
[One paragraph. What the reader understands or can do after this tutorial.]
 
## Files Involved
[Exact paths, verified by CodeGraph search.]
 
## How It Works (High-Level Flow)
[Mermaid diagram or numbered list. Verified by codegraph_callers/callees.]
 
## Step-by-Step Walkthrough
[One section per major step. Code from verified source files only.]
 
## Key Functions and Classes
[Table: Name | File | What It Does — from CodeGraph]
 
## How to Extend or Modify
[Where to add new cases. Which functions to change. Backed by codegraph_impact.]
 
## How to Test It
[Exact test commands. Test files found by codegraph_search.]
 
## Common Mistakes
[What breaks if you do X. Error messages to expect.]
```
 
## Output Location
Save to: `docs/TUTORIALS/tutorial-NN-[short-name].md`
Increment NN from the highest existing tutorial number.
 
## Rules
- Use only exact names from CodeGraph results. No invented names.
- If FastCode is available, always use code_qa before writing the walkthrough.
- Reuse the same session_id for all follow-up questions about the same feature.
```
 
---
 
### Skill 4: security-review
 
```powershell
notepad .claude\skills\security-review\SKILL.md
```
 
```markdown
---
name: security-review
description: >
  Review code for security vulnerabilities using structural graph analysis before
  any file reading. Use when asked to audit security, map attack surface, trace
  source-to-sink flows, review authentication/authorization, or assess a feature
  for security risks. Reports only evidence-backed findings with exact file and
  function references.
---
 
# Security Review Skill
Source: https://github.com/colbymchenry/codegraph | https://github.com/HKUDS/FastCode
 
## Required Workflow
 
### Step 1 — Map the attack surface with CodeGraph
Use `codegraph_search` to find security-sensitive symbols:
- `authenticate`, `authorize`, `checkPermission`, `validateToken`
- Route handlers / controller entry points
- Database query builders (`query`, `execute`, `raw`)
- File system operations (`readFile`, `writeFile`, `upload`)
- Deserialization / parsing (`parse`, `deserialize`, `fromJson`)
- Shell execution (`exec`, `spawn`, `system`, `popen`)
- Crypto / secret handling (`encrypt`, `decrypt`, `hash`, `secret`)
- Outbound HTTP clients
 
For each match, run:
- `codegraph_callers` — who calls this function (trace the input path)
- `codegraph_callees` — what does it call (trace toward sinks)
- `codegraph_impact` — full dependency graph for blast-radius awareness
 
### Step 2 — Semantic reasoning with FastCode
After CodeGraph identifies the structural paths, use `code_qa` to reason about exploitability:
 
```
Use code_qa with:
  repos: ["/path/to/project"]
  query: "Review the authentication and authorization implementation. Are there
          any paths where an unauthenticated user could reach protected resources?
          Trace from the route handler through middleware to the data layer."
```
 
**FastCode escalation triggers in security review:**
- CodeGraph shows a path but you need to assess whether existing controls prevent exploitation
- You need to reason about indirect injection (e.g., second-order SQL injection)
- The question is about logic flaws, not just structural missing controls
- You need to assess whether a dependency is vulnerable to known CVEs
 
**Multi-turn session for thorough review:**
```
# Start a session for the whole security review
Use code_qa: "Give me an overview of all user input entry points and how they are validated."
 
# Follow up with the same session_id
Use code_qa session_id=[id]: "Now focus on the database query paths from those entry points."
 
# Follow up again
Use code_qa session_id=[id]: "Which of those paths is missing parameterization or escaping?"
```
 
### Step 3 — Read only verified files
Read only the files that CodeGraph confirmed are in the suspicious path.
Never read files speculatively in a security review.
 
### Step 4 — Write findings
One finding per vulnerability. Use the exact format below.
 
## Required Finding Format
 
```
### [SEVERITY] [Title]
 
**CWE/OWASP:** CWE-XXX or OWASP Top 10 category (if applicable)
**File:** exact/path/to/file.ext (from CodeGraph search)
**Function:** exactFunctionName() (from CodeGraph search)
**Data Flow:** source_function() → intermediate() → sink_function()
              (traced with codegraph_callers / codegraph_callees)
**Exploit Condition:** exact input or state that triggers the vulnerability
**Existing Controls:** what controls exist and why they do or do not prevent exploitation
**FastCode Analysis:** [if code_qa was used — what semantic reasoning concluded]
**Remediation:** specific code change recommendation
**Patch Suggestion:** [optional — only if safe to provide]
```
 
## Severity Levels
- **CRITICAL** — exploitable without auth, RCE or data exfiltration possible
- **HIGH** — exploitable with normal user access, significant impact
- **MEDIUM** — requires specific conditions, moderate impact
- **LOW** — defense-in-depth issue, minimal direct impact
- **INFO** — observation without direct exploitability
 
## Rules
- Every finding requires a file and function verified by CodeGraph.
- No speculation without structural evidence.
- If a path cannot be fully traced by CodeGraph, mark it `[UNVERIFIED PATH — manual review required]`.
- Report absence of controls explicitly (e.g., "no auth middleware found on this route per codegraph_callers").
- FastCode code_qa is for exploitability reasoning, not for replacing CodeGraph structural tracing.
```
 
---
 
### Skill 5: feature-planner
 
```powershell
notepad .claude\skills\feature-planner\SKILL.md
```
 
```markdown
---
name: feature-planner
description: >
  Plan a new feature implementation by finding an analogous existing pattern,
  identifying the minimal change set with CodeGraph impact analysis, synthesizing
  the approach with FastCode if available, and producing a concrete plan before
  any code is written. Use when asked to implement a feature, add an endpoint,
  extend a module, or refactor a subsystem.
---
 
# Feature Planner Skill
Source: https://github.com/colbymchenry/codegraph | https://github.com/HKUDS/FastCode
 
## Core Rule
Never write code before the plan is approved.
Find the pattern. Verify the impact. Then plan.
 
## Required Workflow
 
### Step 1 — Find the analogous existing feature (CodeGraph)
```
codegraph_search    → find a similar existing feature by symbol name
codegraph_explore   → understand how that feature is structured end-to-end
codegraph_callers   → what calls the analogous feature (integration points)
codegraph_callees   → what the analogous feature depends on (required dependencies)
```
 
The analogous feature shows the correct pattern to follow. Always find one before planning.
 
### Step 2 — Assess the impact (CodeGraph)
```
codegraph_impact    → blast radius of the analogous feature (guides scope)
codegraph_context   → build task context for the new feature
```
 
This identifies the minimal change set — what must be created, what must be modified,
and what must NOT be touched.
 
### Step 3 — Synthesize the approach (FastCode, if available)
After CodeGraph gives structure and impact, use `code_qa` for judgment:
 
```
Use code_qa with:
  repos: ["/path/to/project"]
  query: "I want to add [FEATURE]. I found [ANALOGOUS FEATURE] as the pattern to follow.
          Based on the codebase structure, what is the best approach? What risks should
          I be aware of? Are there any architectural constraints I should respect?"
```
 
If FastCode is available, also ask:
```
Use code_qa session_id=[id]:
  "What tests would I need to write for [FEATURE]?
   What edge cases does the existing [ANALOGOUS FEATURE] handle that I need to replicate?"
```
 
### Step 4 — Produce the plan (present before coding)
 
```markdown
## Feature: [Name]
 
### Summary
[One paragraph.]
 
### Analogous Existing Feature
File: exact/path (from codegraph_search)
Function/class: exactName (from codegraph_search)
Why it's analogous: [reason]
 
### Files to Create
- path/to/new-file.ext — [purpose]
 
### Files to Modify
- path/to/existing-file.ext — [what changes and why, verified by codegraph_impact]
 
### Files NOT to Touch
- [files that might seem relevant but are out of scope — prevents scope creep]
 
### Implementation Steps
1. [Step with exact file and function from CodeGraph]
2. [Step with exact file and function from CodeGraph]
 
### Tests Required
- [test file from codegraph_search] — [what to test]
 
### Risks
- [anything identified by codegraph_impact or FastCode code_qa]
```
 
### Step 5 — Wait for approval
Do not write any code until the user approves the plan.
If changes are requested, revise and re-present.
 
## Rules
- Always find an analogous pattern first using codegraph_search.
- Always run codegraph_impact before deciding what to modify.
- Always list files NOT to touch explicitly.
- FastCode code_qa adds judgment — call it after, not instead of, CodeGraph.
- Never write code before the plan is approved.
```
 
---
 
## Phase 7 — Verify Skills
 
```powershell
claude
```
```
/skills
```
Expected: `repo-navigation`, `documentation-writer`, `tutorial-writer`, `security-review`, `feature-planner`
 
**Test each skill:**
```
/repo-navigation Map the main architecture. Use CodeGraph first. Escalate to FastCode only if needed.
 
/documentation-writer Create docs/ARCHITECTURE.md. Verify all structure with CodeGraph.
Use FastCode code_qa for the narrative overview section only.
 
/tutorial-writer Create docs/TUTORIALS/tutorial-01-main-flow.md for the main execution flow.
 
/security-review Map authentication and authorization. Use codegraph_callers and codegraph_callees
for source-to-sink tracing. Use FastCode code_qa to assess exploitability.
 
/feature-planner Plan how to add request logging to every API endpoint.
Find the analogous pattern with codegraph_search first. Produce a plan before writing code.
```
 
---
 
## Phase 8 — Daily Usage
 
### Start of session
```
/mcp               ← confirm codegraph and fastcode are connected
list_repos         ← confirm project is indexed in FastCode (if available)
list_sessions      ← check for prior sessions to reuse
```
 
### Implementation
```
/feature-planner I need to implement [FEATURE].
Find the analogous pattern with codegraph_search first.
Use FastCode code_qa to assess the architectural approach.
Produce a plan and wait for my approval before writing code.
```
 
### Documentation
```
/documentation-writer Create docs/API_REFERENCE.md.
Use codegraph_explore and codegraph_search for structure.
Use FastCode code_qa for the conceptual overview section.
```
 
### Security
```
/security-review Find all input paths that reach database queries.
Use codegraph_callers and codegraph_impact for structural tracing.
Use FastCode code_qa to assess whether controls prevent exploitation.
```
 
### Update Claude Code (run periodically)
```powershell
winget upgrade Anthropic.ClaudeCode
```
 
---
 
## Troubleshooting
 
**Claude Code PATH issue:** Add `%USERPROFILE%\.local\bin` to user PATH, reopen PowerShell.
 
**CodeGraph Backend: wasm:** Install VS Build Tools, run `npm rebuild better-sqlite3`.
 
**CodeGraph tools not in /mcp:** Run `codegraph init` in project root, restart Claude Code, run `/doctor`.
 
**FastCode not connecting:** Test with `python mcp_server.py` manually; check `.env` has no placeholder values; verify JSON paths use `\\`.
 
**FastCode index stale after refactor:** Call `remove_repo` then `code_qa` to force fresh indexing.
 
**FastCode unavailable:** All skills degrade gracefully. CodeGraph + targeted reads covers the majority of tasks.
 
---
 
## Quick Reference
 
```
TOOLS — WHEN TO USE EACH:
 
codegraph_explore     → First call for any new topic. "What is this?"
codegraph_search      → Find a specific symbol by name. Returns location only.
codegraph_context     → Build task context before implementing or documenting.
codegraph_callers     → Who calls X? Always run before changing a function.
codegraph_callees     → What does X call? Understand internal dependencies.
codegraph_impact      → What breaks if I change X? Always run before refactoring.
 
code_qa               → Semantic Q&A. Use when CodeGraph gives structure but not meaning.
                        Pass repos=[path], query=question, session_id for multi-turn.
list_repos            → See what FastCode has indexed. Run at session start.
list_sessions         → Find prior sessions to reuse. Saves tokens.
get_session           → Review prior analysis before continuing work.
remove_repo           → Force re-index after major refactor.
 
ESCALATION ORDER:
1. CodeGraph (always first — free, local, instant)
2. FastCode code_qa (if CodeGraph can't answer semantically)
3. Read specific files CodeGraph/FastCode identified
4. Targeted grep (last resort, specific string only)
NEVER: broad grep as first exploration step
```
 
Sources:
- FastCode: https://github.com/HKUDS/FastCode
- CodeGraph: https://github.com/colbymchenry/codegraph
- Claude Code Skills: https://code.claude.com/docs/en/skills
- Claude Code Setup: https://code.claude.com/docs/en/setup
 
