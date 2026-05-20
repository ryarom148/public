# Claude Code — Windows Setup: FastCode + CodeGraph + Skills
## Instructions for Claude Code to execute
 
> **How to use:**
> Start Claude Code with `claude`, then say:
> `"Read SETUP_CLAUDE_CODE_WINDOWS.md and execute every phase in order. Stop and report to me if any step fails."`
 
---
 
## Corrections vs common guides
 
Before anything else, three facts confirmed from the CodeGraph GitHub source
(https://github.com/colbymchenry/codegraph) that are frequently wrong in guides:
 
**1. Per-project command is `codegraph init`, not `codegraph install --location=local`.**
The README and npm page both document: install globally once, then run `codegraph init`
in each project. The `--location=local` flag exists in the CHANGELOG for CI/scripting
but `codegraph init` is the canonical per-project command.
 
**2. `codegraph index` is not a separate step after `init`.**
`codegraph init` builds the index as part of initialization.
Re-index after major changes with `codegraph index` (documented in the README).
 
**3. NEVER call `codegraph_explore` or `codegraph_context` from the main Claude Code session.**
The README states this explicitly. These tools return large source-code payloads that
fill the main session context window. They belong inside **Explore subagents** only.
The main session uses only the lightweight lookup tools:
`codegraph_search`, `codegraph_callers`, `codegraph_callees`, `codegraph_impact`.
 
---
 
## Mental Model
 
```
FastCode  = WHAT does this code mean?   (semantic Q&A, costs tokens, use selectively)
CodeGraph = WHERE is everything?        (AST graph, zero tokens, instant, always first)
Claude    = Write it. Explain it. Document it. Test it.
 
Main session CodeGraph tools (lightweight, use directly):
  codegraph_search    → find a symbol by name → returns location only
  codegraph_callers   → who calls function X
  codegraph_callees   → what does function X call
  codegraph_impact    → full blast radius before refactoring
 
Explore-agent-only CodeGraph tools (NEVER in main session):
  codegraph_explore   → large source payload, fills context — spawn Explore agent
  codegraph_context   → large source payload, fills context — spawn Explore agent
 
FastCode tools (always check /mcp first):
  code_qa             → semantic Q&A, multi-turn sessions, multi-repo
  list_repos          → what is indexed
  list_sessions       → prior sessions to reuse
  get_session         → review prior analysis
  remove_repo         → force re-index after major refactor
  delete_session      → clean up stale sessions
```
 
---
 
## Frequency Table
 
| Action | When |
|---|---|
| `npm install -g @colbymchenry/codegraph` | **Once ever** on this machine |
| `npx @colbymchenry/codegraph` (global installer) | **Once ever** — configures Claude Code MCP |
| `codegraph init` | **Once per project** — initializes + indexes |
| `codegraph index` | **After major code changes** — re-indexes |
| `codegraph sync` | **Automatic** — file watcher handles incremental sync |
| `codegraph status` | **Anytime** — check backend and node count |
| `winget upgrade Anthropic.ClaudeCode` | **Periodically** — winget does not auto-update |
 
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
 
**Install anything missing:**
 
```powershell
# Claude Code (if missing)
irm https://claude.ai/install.ps1 | iex
# OR: winget install --id Anthropic.ClaudeCode -e --source winget
 
# Git (required by Claude Code internally)
winget install --id Git.Git -e --source winget
 
# Node.js LTS (required by CodeGraph)
winget install --id OpenJS.NodeJS.LTS -e --source winget
 
# Python 3.12 (required by FastCode)
winget install --id Python.Python.3.12 -e --source winget
```
 
Close and reopen PowerShell after every `winget install`. PATH changes only take effect in new sessions.
 
**If `claude` is not found after install:**
```powershell
[System.Environment]::SetEnvironmentVariable(
  "PATH",
  "$env:USERPROFILE\.local\bin;" + [System.Environment]::GetEnvironmentVariable("PATH","User"),
  "User"
)
# Close and reopen PowerShell, then retry: claude --version
```
 
---
 
## Phase 2 — Install CodeGraph (once ever)
 
Source: https://github.com/colbymchenry/codegraph
 
### Step 2a — Install globally
 
```powershell
npm install -g @colbymchenry/codegraph
codegraph --version
```
 
### Step 2b — Run the global installer once
 
This configures the MCP server in `~/.claude.json`, sets up auto-allow permissions,
and writes global agent instructions. Run it once. Never per-project.
 
```powershell
npx @colbymchenry/codegraph
```
 
When prompted: select **Claude Code**, then **global**.
 
Non-interactive equivalent:
```powershell
codegraph install --target=claude --location=global --yes
```
 
### Step 2c — Verify the global MCP registration
 
```powershell
claude
```
```
/mcp
```
Confirm `codegraph` appears. If not, run `/doctor`.
 
### Step 2d — Fix Windows backend if needed
 
After install, check the backend is native (not the WASM fallback):
 
```powershell
# In any directory:
codegraph status
```
 
If it shows `Backend: wasm` (5–10x slower), fix it:
```powershell
winget install --id Microsoft.VisualStudio.2022.BuildTools -e --source winget
npm rebuild better-sqlite3
# Reopen PowerShell, then:
codegraph status   # must show Backend: native
```
 
---
 
## Phase 3 — Install FastCode (once ever)
 
Source: https://github.com/HKUDS/FastCode
 
FastCode is optional. If unavailable, all skills degrade gracefully to CodeGraph-only.
 
```powershell
# Install uv (fast Python package manager)
pip install uv
uv --version
 
# Clone to a permanent tools location (NOT inside any project)
New-Item -ItemType Directory -Force -Path "C:\tools"
cd C:\tools
git clone https://github.com/HKUDS/FastCode.git
cd FastCode
 
# Read the README before installing — follow it if it differs from below
type README.md
 
# Create venv and install (current v1.0.1)
uv venv --python=3.12
.\.venv\Scripts\Activate.ps1      # Windows: NOT source .venv/bin/activate
uv pip install -r requirements.txt
python -c "import fastcode; print('FastCode OK')"
deactivate
```
 
**Configure API key:**
```powershell
copy env.example .env
notepad .env
```
```env
OPENAI_API_KEY=sk-your-key-here
MODEL=gpt-4o
BASE_URL=https://api.openai.com/v1
 
# For Anthropic Claude API:
# OPENAI_API_KEY=sk-ant-your-key-here
# MODEL=claude-sonnet-4-6
# BASE_URL=https://api.anthropic.com/v1
```
 
**Test the server manually before registering:**
```powershell
cd C:\tools\FastCode
.\.venv\Scripts\Activate.ps1
python mcp_server.py    # should start cleanly; Ctrl+C to stop
deactivate
```
 
**Register with Claude Code:**
```powershell
claude mcp add fastcode -- "C:\tools\FastCode\.venv\Scripts\python.exe" "C:\tools\FastCode\mcp_server.py"
```
 
Or edit `%USERPROFILE%\.claude\claude_desktop_config.json`:
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
 
**Verify:**
```
/mcp    ← both codegraph and fastcode must appear
```
 
---
 
## Phase 4 — Per-Project Bootstrap Script
 
Save this as `setup-claude-project.ps1` in `C:\tools\` or anywhere on your PATH.
Run it once for every new project. Never run it for projects already initialized.
 
```powershell
# setup-claude-project.ps1
# Usage: .\setup-claude-project.ps1 "C:\path\to\my-project"
# Source: https://github.com/colbymchenry/codegraph (init command)
#         https://github.com/HKUDS/FastCode (FastCode usage notes)
 
param(
    [Parameter(Mandatory=$true)]
    [string]$ProjectPath
)
 
# ── Resolve and validate path ────────────────────────────────────────────────
$ProjectPath = Resolve-Path $ProjectPath -ErrorAction Stop
Write-Host ""
Write-Host "=== Claude Code Project Bootstrap ===" -ForegroundColor Cyan
Write-Host "Project : $ProjectPath" -ForegroundColor Cyan
Write-Host ""
 
Set-Location $ProjectPath
 
# ── Step 1: CodeGraph per-project init ───────────────────────────────────────
# codegraph init initializes AND indexes in one step.
# Source: https://github.com/colbymchenry/codegraph README
Write-Host "[1/6] Initializing CodeGraph..." -ForegroundColor Yellow
 
if (Test-Path ".codegraph") {
    Write-Host "      .codegraph/ already exists — running codegraph index to refresh." -ForegroundColor Gray
    codegraph index
} else {
    codegraph init
}
 
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: codegraph init failed. Check 'codegraph status' and 'codegraph --help'." -ForegroundColor Red
    exit 1
}
 
codegraph status
Write-Host ""
 
# ── Step 2: Create .claude directory structure ────────────────────────────────
Write-Host "[2/6] Creating .claude/ project structure..." -ForegroundColor Yellow
 
New-Item -ItemType Directory -Force -Path ".claude"              | Out-Null
New-Item -ItemType Directory -Force -Path ".claude\skills"       | Out-Null
New-Item -ItemType Directory -Force -Path ".claude\skills\repo-navigation"        | Out-Null
New-Item -ItemType Directory -Force -Path ".claude\skills\documentation-writer"   | Out-Null
New-Item -ItemType Directory -Force -Path ".claude\skills\tutorial-writer"        | Out-Null
New-Item -ItemType Directory -Force -Path ".claude\skills\security-review"        | Out-Null
New-Item -ItemType Directory -Force -Path ".claude\skills\feature-planner"        | Out-Null
New-Item -ItemType Directory -Force -Path "docs"                 | Out-Null
New-Item -ItemType Directory -Force -Path "docs\TUTORIALS"       | Out-Null
 
Write-Host ""
 
# ── Step 3: Write CLAUDE.md ───────────────────────────────────────────────────
Write-Host "[3/6] Writing CLAUDE.md..." -ForegroundColor Yellow
 
$claudeMd = @'
# Project Operating Instructions
# Sources:
#   CodeGraph: https://github.com/colbymchenry/codegraph
#   FastCode:  https://github.com/HKUDS/FastCode
 
## CodeGraph Usage Rules
 
**NEVER call `codegraph_explore` or `codegraph_context` from the main session.**
These tools return large source-code payloads that fill the main context window.
Always spawn an Explore subagent for exploration questions.
 
**Main session — lightweight tools only:**
| Tool | Use for |
|---|---|
| `codegraph_search` | Find a symbol by name. Returns location only (no code). |
| `codegraph_callers` | Who calls function X? Run before changing any function. |
| `codegraph_callees` | What does function X call? |
| `codegraph_impact` | Full blast radius. Run before every refactor. |
 
**When spawning an Explore agent**, include this in the prompt:
> This project has CodeGraph initialized (.codegraph/ exists).
> Use `codegraph_explore` as your PRIMARY tool — it returns full source
> code sections in one call. Do NOT re-read files already returned by
> codegraph_explore. Only fall back to grep/glob/Read for files listed
> under "Additional relevant files" if you need more detail.
 
## FastCode Usage Rules
 
Check `/mcp` before using FastCode. If `fastcode` is not listed, skip to CodeGraph.
 
**Use `code_qa` when:**
- CodeGraph answers WHERE but not WHY or HOW (conceptually)
- The question spans multiple repos (pass both paths in `repos=[]`)
- A new developer needs an architecture overview
- Symbol not found in CodeGraph after retry
 
**FastCode MCP tools:**
| Tool | When to call |
|---|---|
| `list_repos` | Session start — confirm repo is indexed |
| `list_sessions` | Session start — find prior sessions to reuse |
| `code_qa` | Semantic Q&A. Pass `session_id` for follow-ups. |
| `get_session` | Review prior analysis before continuing |
| `remove_repo` | After a major refactor — force fresh re-index |
| `delete_session` | Clean up stale sessions |
 
**`code_qa` parameters:**
- `repos`: list of local paths or GitHub URLs (auto-cloned if URL)
- `query`: natural language question
- `session_id`: pass back the returned id for multi-turn follow-ups
- `multi_turn`: default true — uses prior Q&A for context
 
## Tool Decision Rule
 
```
1. Can codegraph_search / callers / callees / impact answer this?
      YES → Use it. Zero tokens. Instant.
       NO → Is FastCode connected? (/mcp)
              YES → code_qa with specific question + repo path
                    Reuse session_id for follow-ups
               NO → Spawn Explore agent with codegraph_explore
                    Or read only the minimal files identified above
```
 
Never start with broad grep across the whole repository.
 
## Documentation Rule
Verify structure with CodeGraph lightweight tools before writing anything.
Use FastCode code_qa for narrative/conceptual sections.
Save docs to /docs/. Save tutorials to /docs/TUTORIALS/.
Include exact source file and function references.
 
## Security Rule
Use codegraph_callers / codegraph_callees for source-to-sink tracing.
Use FastCode code_qa to reason about exploitability of traced paths.
Report only findings with a verified file, function, and data path.
 
## Reindexing
Run `codegraph index` after major code changes.
Run FastCode `remove_repo` then `code_qa` to force fresh semantic re-index.
'@
 
Set-Content -Path "CLAUDE.md" -Value $claudeMd -Encoding UTF8
Write-Host ""
 
# ── Step 4: Write Skills ──────────────────────────────────────────────────────
Write-Host "[4/6] Writing Skills..." -ForegroundColor Yellow
 
# ── repo-navigation ──────────────────────────────────────────────────────────
Set-Content -Path ".claude\skills\repo-navigation\SKILL.md" -Encoding UTF8 -Value @'
---
name: repo-navigation
description: >
  Explore the repository to locate code, trace behavior, map dependencies, or
  prepare context before implementing or documenting. Applies the CodeGraph-first
  rule: lightweight main-session tools first, Explore subagent for source payload,
  FastCode for semantic questions. Never starts with broad grep.
---
 
# Repo Navigation Skill
# Source: https://github.com/colbymchenry/codegraph
#         https://github.com/HKUDS/FastCode
 
## Critical CodeGraph Rule (from official README)
 
NEVER call `codegraph_explore` or `codegraph_context` in the main session.
These return large source payloads that fill the context window.
 
Main session: use ONLY the lightweight tools below.
Source payload: spawn an Explore subagent.
 
## Step 1 — Lightweight CodeGraph lookups (main session, always first)
 
| Tool | Question it answers |
|---|---|
| `codegraph_search` | Where is symbol X? (file + line, no code) |
| `codegraph_callers` | What calls function X? |
| `codegraph_callees` | What does function X call? |
| `codegraph_impact` | What breaks if I change X? |
 
Use these to identify which files matter. Then proceed to Step 2 or 3.
 
## Step 2 — Source exploration (spawn Explore subagent)
 
When you need source code, spawn an Explore subagent with this instruction:
 
> This project has CodeGraph initialized (.codegraph/ exists).
> Use `codegraph_explore` as your PRIMARY tool.
> Rules:
> 1. Follow the explore call budget in the tool description.
> 2. Do NOT re-read files that codegraph_explore already returned.
> 3. Only fall back to grep/glob/Read for "Additional relevant files"
>    if you need more detail, or if codegraph returned no results.
 
## Step 3 — FastCode semantic scouting (if available, after Steps 1-2)
 
Check /mcp. If `fastcode` is listed, use `code_qa` when:
- The question requires understanding meaning, not just structure
- codegraph_search returns nothing after retry with shorter name
- The question spans two repos (pass both in `repos=[]`)
- A new developer needs a conceptual architecture overview
 
**code_qa usage:**
```
repos:      ["/absolute/path/to/project"]   # or GitHub URL — auto-cloned
query:      "Specific semantic question"
session_id: "prior-id"                      # omit for new session; reuse for follow-ups
multi_turn: true                            # default
```
 
**Session management:**
1. Call `list_repos` — confirm repo is indexed
2. Call `list_sessions` — find prior sessions to reuse
3. Call `code_qa` — save the returned session_id
4. All follow-up questions: pass the same session_id
 
**Re-index after major refactor:**
1. Call `remove_repo` with the repo path
2. Call `code_qa` — FastCode re-indexes automatically
 
## Step 4 — Targeted file reads and grep (last resort)
 
Read only files identified by Steps 1-3.
Use grep only for a specific known string — never as exploration.
 
## If Both Tools Unavailable
 
Report: "CodeGraph and FastCode unavailable — run /mcp to check status."
Ask for explicit permission before falling back to directory browsing.
 
## Output Format
 
1. CodeGraph lightweight lookup results (tools called, symbols found)
2. Explore subagent summary (if spawned)
3. FastCode code_qa findings (if used, include session_id)
4. Files identified as relevant and why
5. Files excluded and why
6. Recommended next action
'@
 
# ── documentation-writer ─────────────────────────────────────────────────────
Set-Content -Path ".claude\skills\documentation-writer\SKILL.md" -Encoding UTF8 -Value @'
---
name: documentation-writer
description: >
  Create or update accurate project documentation grounded in verified repository
  structure. Use when asked to write architecture docs, API references, workflow
  guides, or getting-started documentation. Verifies all structure with CodeGraph
  lightweight tools before writing. Uses FastCode code_qa for narrative sections.
---
 
# Documentation Writer Skill
# Source: https://github.com/colbymchenry/codegraph
#         https://github.com/HKUDS/FastCode
 
## Required Workflow
 
### Step 1 — Lightweight CodeGraph verification (main session)
 
Before writing, use the lightweight tools to verify what exists:
- `codegraph_search` → confirm exact symbol names and file paths
- `codegraph_callers` / `codegraph_callees` → verify described flows
- `codegraph_impact` → verify dependency relationships stated in the doc
 
Do not document any file, function, or flow that these tools cannot confirm.
 
### Step 2 — Source detail (Explore subagent)
 
For sections requiring source code detail, spawn an Explore subagent:
 
> This project has CodeGraph initialized. Use codegraph_explore as PRIMARY tool.
> Question: [specific structural question for the doc section being written]
 
### Step 3 — Conceptual synthesis (FastCode, when available)
 
Use `code_qa` for sections requiring narrative explanation:
```
repos:  ["/path/to/project"]
query:  "Explain [module/feature] for a developer reading the documentation.
         What is its purpose, how does it fit the architecture, and what
         are the key extension points?"
```
Save the session_id. Use it for follow-up questions about specific sections.
 
### Step 4 — Write from verified evidence only
 
- File paths: from codegraph_search
- Function/class names: exact names from codegraph_search
- Flows: verified by codegraph_callers / codegraph_callees
- Narrative: from FastCode code_qa when used
- Anything unverified: mark `[UNVERIFIED — needs review]`
 
## Output Paths
 
| Document | Path |
|---|---|
| Architecture overview | `docs/ARCHITECTURE.md` |
| Getting started | `docs/GETTING_STARTED.md` |
| Core workflows | `docs/CORE_WORKFLOWS.md` |
| API reference | `docs/API_REFERENCE.md` |
| Feature development guide | `docs/FEATURE_DEVELOPMENT_GUIDE.md` |
| Security model | `docs/SECURITY_MODEL.md` |
 
## Required Structure Per Document
 
1. Purpose — what this covers and who it is for
2. Relevant source files — exact paths from codegraph_search
3. Main flow — entry to output, verified by callers/callees
4. Key functions/classes — exact names, what they do
5. Dependencies — verified by CodeGraph
6. Extension points — where to add new behavior
7. Testing notes — test files from codegraph_search
8. Mermaid diagram — when flow has more than 3 steps
'@
 
# ── tutorial-writer ──────────────────────────────────────────────────────────
Set-Content -Path ".claude\skills\tutorial-writer\SKILL.md" -Encoding UTF8 -Value @'
---
name: tutorial-writer
description: >
  Generate developer tutorials grounded in actual codebase behavior. Use when asked
  to create a step-by-step tutorial, walkthrough, or how-to guide for a feature,
  flow, or pattern. Uses CodeGraph lightweight tools for structure, Explore subagent
  for source, and FastCode code_qa for the narrative walkthrough.
---
 
# Tutorial Writer Skill
# Source: https://github.com/colbymchenry/codegraph
#         https://github.com/HKUDS/FastCode
 
## Required Workflow
 
### Step 1 — Locate the feature (lightweight CodeGraph)
 
```
codegraph_search    → find entry point by symbol name
codegraph_callers   → trace the call chain backward (who triggers this?)
codegraph_callees   → trace the call chain forward (what does it do?)
codegraph_impact    → understand the feature's scope and dependencies
```
 
### Step 2 — Get source detail (Explore subagent)
 
Spawn an Explore subagent:
> This project has CodeGraph initialized. Use codegraph_explore.
> Question: Trace the full implementation of [feature] from entry point
> [symbol from Step 1] to output. Return the relevant source sections.
 
### Step 3 — Synthesize the narrative (FastCode, if available)
 
```
repos:      ["/path/to/project"]
query:      "Walk me through how [feature] works step by step for a developer
             writing a tutorial. Focus on the flow from user action to output,
             the key decisions made, and common mistakes."
session_id: (save for follow-ups)
```
 
Follow-up in same session:
```
session_id: [same id]
query:      "What are the edge cases and error handling for [feature]?"
```
 
### Step 4 — Write using exact verified names
 
Use only function/class names from codegraph_search results.
Use only file paths verified by CodeGraph.
Base the narrative on FastCode code_qa analysis.
 
## Required Tutorial Structure
 
```markdown
# Tutorial: [Title]
 
## What You Will Learn
[One paragraph.]
 
## Files Involved
[Exact paths from codegraph_search.]
 
## How It Works (High-Level Flow)
[Mermaid diagram — verified by codegraph_callers/callees.]
 
## Step-by-Step Walkthrough
[One section per major step. Code from Explore subagent only.]
 
## Key Functions and Classes
[Table: Name | File | What It Does — from codegraph_search]
 
## How to Extend or Modify
[From codegraph_impact — what to change and where.]
 
## How to Test It
[Test files from codegraph_search. Exact test commands.]
 
## Common Mistakes
[What breaks. Error messages to expect.]
```
 
## Output Location
 
Save to: `docs/TUTORIALS/tutorial-NN-[short-name].md`
Increment NN from the highest existing tutorial number.
'@
 
# ── security-review ──────────────────────────────────────────────────────────
Set-Content -Path ".claude\skills\security-review\SKILL.md" -Encoding UTF8 -Value @'
---
name: security-review
description: >
  Review code for security vulnerabilities using structural graph analysis before
  any file reading. Use when asked to audit security, map attack surface, trace
  source-to-sink data flows, or assess a feature for security risks. Reports only
  findings backed by exact file, function, and traced data path.
---
 
# Security Review Skill
# Source: https://github.com/colbymchenry/codegraph
#         https://github.com/HKUDS/FastCode
 
## Required Workflow
 
### Step 1 — Map attack surface (lightweight CodeGraph, main session)
 
Use `codegraph_search` to find security-sensitive symbols:
- auth: `authenticate`, `authorize`, `checkPermission`, `validateToken`, `session`
- routes: controller/handler entry points
- database: `query`, `execute`, `raw`, `prepare`
- filesystem: `readFile`, `writeFile`, `upload`, `unlink`
- parsing: `parse`, `deserialize`, `fromJson`, `eval`
- shell: `exec`, `spawn`, `system`, `popen`
- crypto: `encrypt`, `decrypt`, `hash`, `secret`, `key`
- http: outbound client calls
For every match:
- `codegraph_callers` → trace who delivers input to this function (source)
- `codegraph_callees` → trace what this function passes input to (sink)
- `codegraph_impact` → full dependency graph for blast-radius scope
### Step 2 — Source detail (Explore subagent)
 
Spawn Explore subagent for suspicious paths identified in Step 1:
> This project has CodeGraph initialized. Use codegraph_explore.
> Question: Show the implementation of [function] and its immediate callers
> and callees. I am tracing a potential [vulnerability type] path.
 
### Step 3 — Exploitability reasoning (FastCode, if available)
 
After structural tracing, use `code_qa` to reason about exploitability:
```
repos:  ["/path/to/project"]
query:  "I found this data flow: [source] → [intermediate] → [sink].
         Are there existing controls that prevent exploitation?
         What exact input would trigger this path?
         Is there sanitization, parameterization, or auth before the sink?"
```
 
Multi-turn for thorough reviews:
```
# Session 1: overview
code_qa: "Give me all user input entry points and how they are validated."
 
# Session 2 (same session_id): focus
code_qa: "Of those entry points, which reach database query functions without
          parameterization?"
 
# Session 3 (same session_id): confirm
code_qa: "For path [X], does the existing auth middleware cover this route?"
```
 
### Step 4 — Write findings (evidence-only)
 
## Required Finding Format
 
```
### [SEVERITY] [Title]
 
CWE/OWASP:   CWE-XXX (if applicable)
File:        exact/path/file.ext  (from codegraph_search)
Function:    exactFunctionName()  (from codegraph_search)
Data Flow:   source_fn() → intermediate() → sink_fn()
             (traced with codegraph_callers / codegraph_callees)
Exploit:     exact input or state that triggers it
Controls:    what exists and why it does/does not prevent exploitation
FastCode:    [if code_qa was used — semantic reasoning conclusion]
Fix:         specific code change
Patch:       [optional — only if safe to provide]
```
 
## Severity: CRITICAL / HIGH / MEDIUM / LOW / INFO
 
## Rules
- Every finding: file + function verified by codegraph_search.
- Unverifiable paths: mark `[UNVERIFIED PATH — manual review required]`.
- Missing controls: state explicitly ("no auth middleware on this route per codegraph_callers").
- FastCode code_qa: for exploitability reasoning, not replacing CodeGraph tracing.
'@
# ── feature-planner ──────────────────────────────────────────────────────────
Set-Content -Path ".claude\skills\feature-planner\SKILL.md" -Encoding UTF8 -Value @'
---
name: feature-planner
description: >
  Plan a new feature by finding the analogous existing pattern with CodeGraph,
  assessing blast radius with codegraph_impact, synthesizing the approach with
  FastCode, and producing a concrete plan before any code is written. Never writes
  code before the plan is approved.
---
 
# Feature Planner Skill
# Source: https://github.com/colbymchenry/codegraph
#         https://github.com/HKUDS/FastCode
 
## Core Rule
Never write code before the plan is approved.
 
## Required Workflow
 
### Step 1 — Find the analogous pattern (lightweight CodeGraph, main session)
 
```
codegraph_search  → find a similar existing feature by symbol name
codegraph_callers → what calls the analogous feature (integration points)
codegraph_callees → what the analogous feature depends on
codegraph_impact  → blast radius of the analogous feature (guides scope estimate)
```
 
The analogous feature defines the correct pattern to follow.
Always find one before planning.
 
### Step 2 — Get implementation detail (Explore subagent)
 
Spawn Explore subagent:
> This project has CodeGraph initialized. Use codegraph_explore.
> Question: Show the full implementation of [analogous feature] including
> its entry point, service layer, data layer, and tests.
 
### Step 3 — Synthesize approach (FastCode, if available)
 
```
repos:  ["/path/to/project"]
query:  "I want to add [FEATURE]. I found [ANALOGOUS FEATURE] as the pattern.
         What is the best approach? What architectural constraints apply?
         What risks should I be aware of?"
```
 
Follow-up (same session_id):
```
query:  "What tests would I need? What edge cases does [ANALOGOUS FEATURE]
         handle that I must replicate for [NEW FEATURE]?"
```
 
### Step 4 — Produce the plan (present before coding)
 
```markdown
## Feature: [Name]
 
### Summary
[One paragraph.]
 
### Analogous Existing Feature
File:     exact/path  (from codegraph_search)
Symbol:   exactName   (from codegraph_search)
Why analogous: [reason]
 
### Files to Create
- path/to/new-file.ext — [purpose]
 
### Files to Modify
- path/to/file.ext — [what changes, verified by codegraph_impact]
 
### Files NOT to Touch
- [list — prevents scope creep]
 
### Implementation Steps
1. [Step with exact file + function from CodeGraph]
2. [Step with exact file + function from CodeGraph]
 
### Tests Required
- [test file from codegraph_search] — [what to test]
 
### Risks
- [from codegraph_impact + FastCode code_qa]
```
 
### Step 5 — Wait for approval
 
Do not write any code until the user approves the plan.
If changes are requested, revise and re-present.
'@
 
Write-Host "   Skills written." -ForegroundColor Green
Write-Host ""
 
# ── Step 5: Optional — detect language for codegraph config ──────────────────
Write-Host "[5/6] Checking .codegraph/config.json..." -ForegroundColor Yellow
 
$configPath = ".codegraph\config.json"
if (Test-Path $configPath) {
    Write-Host "   config.json exists. Review it to confirm language/exclude settings:" -ForegroundColor Gray
    Get-Content $configPath
} else {
    Write-Host "   config.json not found — codegraph init should have created it." -ForegroundColor Gray
    Write-Host "   If missing, create .codegraph\config.json with your language settings." -ForegroundColor Gray
}
Write-Host ""
 
# ── Step 6: Summary ───────────────────────────────────────────────────────────
Write-Host "[6/6] Done. Summary:" -ForegroundColor Green
Write-Host ""
Write-Host "  Project  : $ProjectPath" -ForegroundColor White
Write-Host "  CodeGraph: initialized (.codegraph/ exists)" -ForegroundColor White
Write-Host "  CLAUDE.md: written" -ForegroundColor White
Write-Host "  Skills   : repo-navigation, documentation-writer, tutorial-writer," -ForegroundColor White
Write-Host "             security-review, feature-planner" -ForegroundColor White
Write-Host "  Docs     : docs/ and docs/TUTORIALS/ created" -ForegroundColor White
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Review and customize CLAUDE.md for this project's specific rules." -ForegroundColor Cyan
Write-Host "  2. Review .codegraph\config.json — add language/exclude settings if needed." -ForegroundColor Cyan
Write-Host "  3. Run: claude" -ForegroundColor Cyan
Write-Host "  4. Inside Claude Code: /mcp  (confirm codegraph and fastcode listed)" -ForegroundColor Cyan
Write-Host "  5. Inside Claude Code: /skills  (confirm all 5 skills listed)" -ForegroundColor Cyan
Write-Host ""
Write-Host "Re-index after major code changes: codegraph index" -ForegroundColor Gray
Write-Host "Update Claude Code periodically:   winget upgrade Anthropic.ClaudeCode" -ForegroundColor Gray
```
 
Save as `C:\tools\setup-claude-project.ps1`.
 
**Usage:**
```powershell
.\setup-claude-project.ps1 "C:\repo\my-project"
```
 
---
 
## Phase 5 — Verify After Bootstrap
 
Inside Claude Code:
 
```
/mcp      ← codegraph and fastcode must be listed
/skills   ← repo-navigation, documentation-writer, tutorial-writer,
             security-review, feature-planner must appear
```
 
Test the correct exploration pattern:
```
Use codegraph_search to find the main entry point of this project.
Then spawn an Explore agent to explain what that entry point does.
Do NOT call codegraph_explore from this main session.
```
 
---
 
## Daily Usage Reference
 
### Start of session
```
/mcp                    ← verify both servers
list_repos              ← confirm FastCode has this repo indexed
list_sessions           ← check for prior sessions to reuse
```
 
### Explore (correct pattern)
```
# Main session: lightweight lookup
Use codegraph_search to find [symbol].
Use codegraph_callers to see what calls it.
 
# Then: source detail via Explore subagent
Spawn an Explore agent: use codegraph_explore to explain how [feature] works.
```
 
### Feature work
```
/feature-planner I need to implement [FEATURE].
Find the analogous pattern with codegraph_search first.
Use FastCode code_qa for architectural judgment if available.
Produce a plan and wait for my approval before writing code.
```
 
### After large refactor
```powershell
codegraph index                    # re-index the graph
# Inside Claude Code:
# Use remove_repo then code_qa     # re-index FastCode semantic layer
```
 
### Update Claude Code
```powershell
winget upgrade Anthropic.ClaudeCode   # run periodically
```
 
---
 
## Troubleshooting
 
| Problem | Fix |
|---|---|
| `claude` not found | Add `%USERPROFILE%\.local\bin` to PATH, reopen PowerShell |
| `codegraph status` shows `Backend: wasm` | Install VS Build Tools + `npm rebuild better-sqlite3` |
| CodeGraph tools not in `/mcp` | Run `codegraph init` in project root, restart Claude Code, run `/doctor` |
| FastCode not connecting | Test `python mcp_server.py` manually; check `.env` has real values; verify JSON uses `\\` |
| FastCode stale after refactor | Call `remove_repo` then `code_qa` to trigger fresh indexing |
| `codegraph init` says already initialized | Run `codegraph index` instead to refresh |
 
---
 
## Quick Reference Card
 
```
ONCE EVER (this machine):
  npm install -g @colbymchenry/codegraph
  npx @colbymchenry/codegraph            ← global MCP config for Claude Code
  git clone FastCode + install deps      ← FastCode setup
 
PER NEW PROJECT:
  .\setup-claude-project.ps1 "C:\path\to\project"
 
AFTER MAJOR CODE CHANGES:
  codegraph index                        ← refresh CodeGraph graph
  remove_repo + code_qa                  ← refresh FastCode semantic index
 
CODEGRAPH TOOL RULES (from official README):
  Main session ONLY:   codegraph_search, codegraph_callers,
                       codegraph_callees, codegraph_impact
  Explore agent ONLY:  codegraph_explore, codegraph_context
 
FASTCODE TOOLS:
  list_repos      → what is indexed (run at session start)
  list_sessions   → prior sessions to reuse (run at session start)
  code_qa         → semantic Q&A; pass session_id for multi-turn
  get_session     → review prior analysis
  remove_repo     → force fresh re-index after refactor
  delete_session  → clean up stale sessions
 
DECISION RULE:
  1. codegraph_search / callers / callees / impact  (free, instant, always first)
  2. Explore subagent with codegraph_explore        (when source code needed)
  3. FastCode code_qa                               (when meaning/semantics needed)
  4. Read specific files                            (only what was identified above)
  5. Targeted grep                                  (last resort, specific string only)
  NEVER: broad grep as first step
```
 
Sources:
- CodeGraph: https://github.com/colbymchenry/codegraph
- FastCode: https://github.com/HKUDS/FastCode
- Claude Code Skills: https://code.claude.com/docs/en/skills
- Claude Code Setup: https://code.claude.com/docs/en/setup
