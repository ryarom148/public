# Claude Code — Windows Setup
## FastCode + CodeGraph + Skills (MCP and No-MCP paths)
 
> **How to use:** Drop this file in your project or tools folder.
> Start Claude Code with `claude`, then say:
> "Read SETUP_CLAUDE_CODE_WINDOWS.md and execute every phase in order.
>  Stop and ask me before continuing if any step fails."
 
---
 
## Section 0 — Mental Model and Sync Behaviour
 
### What each tool does
 
```
CodeGraph = WHERE is everything? HOW is it connected?
            AST graph, SQLite, zero tokens, 100% local. Always use first.
            Source: https://github.com/colbymchenry/codegraph
 
FastCode  = WHAT does this code mean?
            Semantic Q&A using vector embeddings + BM25 + call graphs.
            LLM-powered. Costs tokens. Use when structure is not enough.
            Source: https://github.com/HKUDS/FastCode
 
Claude    = Write it. Explain it. Document it. Test it.
```
 
### How each tool keeps its index current — read this carefully
 
| Tool | How sync works | What to do after `git pull` or major changes |
|---|---|---|
| **CodeGraph MCP** | Built-in file watcher inside the MCP server process. Uses native `ReadDirectoryChangesW` on Windows. Debounced 2-second window. Runs automatically while Claude Code is open. Zero config. | Nothing — auto-handled |
| **CodeGraph No-MCP** | No watcher. No MCP server process running, so no watcher process either. | Run `codegraph index`. A `PostToolUse` hook handles per-edit sync. |
| **FastCode (MCP or CLI)** | No file watcher. No incremental updates. Index is stored as flat files (`.faiss`, `_metadata.pkl`, `_bm25.pkl`, `_graphs.pkl`). Built once on first `code_qa` call. **Stays frozen until you manually clear it.** | MCP: call `remove_repo`, then `code_qa` re-indexes. CLI: delete the index files, then re-run query. |
 
**The practical rule for FastCode:** After any `git pull` that changes significant code, FastCode's semantic index is stale. It does not know this. You must trigger a re-index manually. This is documented explicitly in CLAUDE.md below and in the skills.
 
### Hooks — what is actually needed
 
Only one hook is justified:
 
| Hook | Mode | Why |
|---|---|---|
| `PostToolUse → codegraph sync` | **No-MCP only** | Replaces the absent file watcher so per-edit changes stay indexed |
| Nothing for CodeGraph MCP | MCP | Watcher is already running inside the server process |
| Nothing for FastCode | Either | FastCode has no incremental sync at all — hooks cannot help |
 
### CodeGraph session split — from the official README
 
Never call `codegraph_explore` or `codegraph_context` from the main
Claude Code session. They return large source payloads that fill context.
 
```
Main session — lightweight lookups only:
  codegraph_search   → find a symbol by name, returns file + line
  codegraph_callers  → who calls function X
  codegraph_callees  → what does function X call
  codegraph_impact   → full blast radius before any change
 
Explore subagent only — never call from main session:
  codegraph_explore  → full source sections, spawn as subagent
  codegraph_context  → full task context, spawn as subagent
```
 
### Tool decision rule — apply before every query
 
```
1. Can codegraph_search / callers / callees / impact answer this?
      YES → Use it. Zero tokens. Always first.
       NO → Need source code?
              YES → Spawn Explore subagent with codegraph_explore
               NO → Need semantic understanding?
                     MCP available → code_qa (FastCode)
                     No-MCP → python main.py query (FastCode CLI)
                     Neither → read only files already identified above
Never start with broad grep.
```
 
### Frequency table
 
| Action | Frequency |
|---|---|
| `npm install -g @colbymchenry/codegraph` | Once ever on this machine |
| `npx @colbymchenry/codegraph` | Once ever — writes global MCP config |
| `codegraph init -i` | Once per new project |
| `codegraph index` | After major code changes (No-MCP mode) |
| FastCode clone + venv | Once ever on this machine |
| FastCode `remove_repo` + `code_qa` | After `git pull` or major refactor |
| `winget upgrade Anthropic.ClaudeCode` | Periodically — no auto-update |
 
---
 
## Section 1 — Determine Your Mode
 
Open PowerShell, start Claude Code, and check:
 
```
/mcp
```
 
**Mode A — MCP** (servers appear after setup): Full integration. Follow MCP path.
 
**Mode B — No-MCP** (policy blocks MCP, or `/mcp` shows nothing):
Both tools expose CLI interfaces. Claude calls them via the `Bash` tool.
A single hook handles the absent watcher for CodeGraph.
 
**Mode C — Misconfigured** (listed but tools missing):
Run `/doctor`. Confirm `codegraph init -i` was run in the project.
Restart Claude Code.
 
---
 
## Section 2 — Prerequisites
 
```powershell
claude --version && git --version && node --version && npm --version && python --version
```
 
Install anything missing. **Close and reopen PowerShell after each:**
 
```powershell
# Claude Code
irm https://claude.ai/install.ps1 | iex
# or: winget install --id Anthropic.ClaudeCode -e --source winget
 
# Git (used internally by Claude Code on Windows)
winget install --id Git.Git -e --source winget
 
# Node.js LTS (required by CodeGraph)
winget install --id OpenJS.NodeJS.LTS -e --source winget
 
# Python 3.12 (required by FastCode)
winget install --id Python.Python.3.12 -e --source winget
```
 
If `claude` is not found after install:
```powershell
[System.Environment]::SetEnvironmentVariable(
  "PATH",
  "$env:USERPROFILE\.local\bin;" +
  [System.Environment]::GetEnvironmentVariable("PATH","User"),
  "User"
)
# Close and reopen PowerShell, then retry
```
 
---
 
## Section 3 — Install CodeGraph (once ever)
 
Source: https://github.com/colbymchenry/codegraph
 
```powershell
npm install -g @colbymchenry/codegraph
codegraph --version
 
# Run global interactive installer — writes MCP config to ~/.claude.json
# and sets auto-allow permissions. Run ONCE. Never per project.
npx @colbymchenry/codegraph
# When prompted: Claude Code → global
```
 
Non-interactive (scripting/CI):
```powershell
codegraph install --target=claude --location=global --yes
```
 
**Verify backend is native** (not the 5–10x slower WASM fallback):
```powershell
codegraph status    # must show Backend: native
```
 
If it shows `Backend: wasm`:
```powershell
winget install --id Microsoft.VisualStudio.2022.BuildTools -e --source winget
# Close and reopen PowerShell, then:
npm rebuild better-sqlite3
codegraph status    # must now show Backend: native
```
 
**Manual MCP config** (if the installer did not write it):
Open `%USERPROFILE%\.claude.json` and add:
```json
{
  "mcpServers": {
    "codegraph": {
      "type": "stdio",
      "command": "codegraph",
      "args": ["serve", "--mcp"]
    }
  }
}
```
 
**Verify inside Claude Code:**
```
/mcp    ← codegraph must appear
```
 
---
 
## Section 4 — Install FastCode (once ever)
 
Source: https://github.com/HKUDS/FastCode
 
FastCode is optional. All skills degrade gracefully without it.
 
```powershell
pip install uv
 
New-Item -ItemType Directory -Force -Path "C:\tools"
cd C:\tools
git clone https://github.com/HKUDS/FastCode.git
cd FastCode
 
# Read README before installing — follow it if it differs from below (v1.0.1)
type README.md
 
uv venv --python=3.12
.\.venv\Scripts\Activate.ps1    # Windows — NOT source .venv/bin/activate
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
# Any OpenAI-compatible provider
OPENAI_API_KEY=sk-your-key-here
MODEL=gpt-4o
BASE_URL=https://api.openai.com/v1
```
 
**Test MCP server:**
```powershell
cd C:\tools\FastCode
.\.venv\Scripts\Activate.ps1
python mcp_server.py    # must start without errors; Ctrl+C to stop
deactivate
```
 
**Register MCP server:**
```powershell
claude mcp add fastcode -- `
  "C:\tools\FastCode\.venv\Scripts\python.exe" `
  "C:\tools\FastCode\mcp_server.py"
```
 
Or edit `%USERPROFILE%\.claude\claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "fastcode": {
      "command": "C:\\tools\\FastCode\\.venv\\Scripts\\python.exe",
      "args":    ["C:\\tools\\FastCode\\mcp_server.py"],
      "env": {
        "MODEL":          "gpt-4o",
        "BASE_URL":       "https://api.openai.com/v1",
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
 
## Section 5 — No-MCP Integration (skip if MCP works)
 
When MCP is blocked, Claude calls both tools via the `Bash` tool.
 
### CodeGraph CLI
 
Documented in https://github.com/colbymchenry/codegraph/blob/main/CLAUDE.md:
```powershell
codegraph query "UserService"             # find symbol — file + line
codegraph files "authentication"          # find relevant files
codegraph context "add payment retry"     # full context for a task
codegraph affected "src/auth/session.ts"  # blast radius of a file
codegraph index                           # full re-index
codegraph sync                            # incremental sync (per-edit)
```
 
### FastCode CLI
 
From the FastCode README:
```powershell
cd C:\tools\FastCode
.\.venv\Scripts\Activate.ps1
python main.py query --repo-path "C:\path\to\project" --query "How does auth work?"
```
 
### FastCode re-index after `git pull` (No-MCP)
 
FastCode stores its index as flat files alongside the source code.
After a significant `git pull`, delete them and re-run the query:
```powershell
# FastCode stores index files named like:
#   <repo-name>_metadata.pkl
#   <repo-name>_bm25.pkl
#   <repo-name>_graphs.pkl
#   <repo-name>.faiss
# Delete them (they are inside the FastCode directory):
cd C:\tools\FastCode
Get-ChildItem -Name "*.pkl","*.faiss" | Remove-Item
# Next query call re-indexes automatically
```
 
### The one hook that is actually needed (No-MCP only)
 
Source: https://code.claude.com/docs/en/hooks-guide
 
The CodeGraph MCP server has a built-in file watcher. When MCP is blocked,
no server process runs, so no watcher runs. A `PostToolUse` hook on
`Edit|Write` calls `codegraph sync` to replace the absent watcher.
 
**FastCode needs no hook** — it has no incremental sync at all.
Re-indexing FastCode is always a manual step regardless of mode.
 
Add to `.claude/settings.json` **(No-MCP only)**:
```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [{ "type": "command", "command": "codegraph sync" }]
      }
    ]
  }
}
```
 
---
 
## Section 6 — Per-Project Bootstrap Script
 
Save as `C:\tools\setup-claude-project.ps1`.
Run once per new project.
 
```powershell
# setup-claude-project.ps1
# Usage:
#   .\setup-claude-project.ps1 "C:\path\to\project"         (MCP mode)
#   .\setup-claude-project.ps1 "C:\path\to\project" -NoMCP  (No-MCP mode)
#
# Sources:
#   CodeGraph : https://github.com/colbymchenry/codegraph
#   FastCode  : https://github.com/HKUDS/FastCode
#   Hooks     : https://code.claude.com/docs/en/hooks-guide
 
param(
    [Parameter(Mandatory=$true)][string]$ProjectPath,
    [switch]$NoMCP
)
 
$ProjectPath = Resolve-Path $ProjectPath -ErrorAction Stop
Write-Host ""
Write-Host "=== Claude Code Project Bootstrap ===" -ForegroundColor Cyan
Write-Host "Project : $ProjectPath"
Write-Host "Mode    : $(if ($NoMCP) { 'No-MCP (Bash CLI)' } else { 'MCP' })"
Write-Host ""
Set-Location $ProjectPath
 
# ── 1. CodeGraph init ─────────────────────────────────────────────────────────
Write-Host "[1/5] CodeGraph init..." -ForegroundColor Yellow
 
if (Test-Path ".codegraph") {
    Write-Host "      .codegraph/ exists — running codegraph index to refresh."
    codegraph index
} else {
    # codegraph init -i initializes AND indexes in one step
    # Source: https://github.com/colbymchenry/codegraph README Quick Start
    codegraph init -i
}
 
if ($LASTEXITCODE -ne 0) {
    Write-Error "codegraph init failed. Run 'codegraph status' and 'codegraph --help'."
    exit 1
}
codegraph status
Write-Host ""
 
# ── 2. Directory structure ────────────────────────────────────────────────────
Write-Host "[2/5] Creating project structure..." -ForegroundColor Yellow
 
foreach ($d in @(
    ".claude", ".claude\skills",
    ".claude\skills\repo-navigation",
    ".claude\skills\documentation-writer",
    ".claude\skills\tutorial-writer",
    ".claude\skills\security-review",
    ".claude\skills\feature-planner",
    "docs", "docs\TUTORIALS"
)) { New-Item -ItemType Directory -Force -Path $d | Out-Null }
 
Write-Host ""
 
# ── 3. .claude/settings.json ─────────────────────────────────────────────────
Write-Host "[3/5] Writing .claude/settings.json..." -ForegroundColor Yellow
 
# MCP mode : CodeGraph MCP server has a built-in file watcher — no sync hook needed.
#            Only the post-compaction re-injection hook is useful.
#            FastCode has no incremental sync — no hook can help.
#            Source: https://github.com/colbymchenry/codegraph README (Auto-Sync)
#
# No-MCP   : No MCP server process means no watcher process.
#            PostToolUse hook calls codegraph sync to fill that gap.
#            FastCode still needs no hook — re-index is always manual.
 
$settingsJson = if ($NoMCP) {
@'
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [{ "type": "command", "command": "codegraph sync" }]
      }
    ],
    "SessionStart": [
      {
        "matcher": "compact",
        "hooks": [
          {
            "type": "command",
            "command": "echo 'No-MCP. CodeGraph CLI: codegraph query/files/context/affected. FastCode CLI: cd C:\\tools\\FastCode && .venv\\Scripts\\python.exe main.py query --repo-path FULL_PATH --query QUESTION. After git pull: delete .pkl/.faiss files in C:\\tools\\FastCode then re-run query.'"
          }
        ]
      }
    ]
  }
}
'@
} else {
@'
{
  "hooks": {
    "SessionStart": [
      {
        "matcher": "compact",
        "hooks": [
          {
            "type": "command",
            "command": "echo 'CodeGraph MCP active. Main session: codegraph_search/callers/callees/impact only. Explore subagent: codegraph_explore. FastCode: code_qa. After git pull: call remove_repo then code_qa to re-index FastCode.'"
          }
        ]
      }
    ]
  }
}
'@
}
 
Set-Content -Path ".claude\settings.json" -Value $settingsJson -Encoding UTF8
Write-Host ""
 
# ── 4. CLAUDE.md ──────────────────────────────────────────────────────────────
Write-Host "[4/5] Writing CLAUDE.md..." -ForegroundColor Yellow
 
$integrationBlock = if ($NoMCP) {
@'
## Integration Mode: No-MCP (Bash CLI)
MCP is not available. Use the Bash tool to call both tools directly.
 
### CodeGraph CLI (always first — zero tokens)
Run from project root before reading any file:
  codegraph query "symbol"             → find symbol, returns file + line
  codegraph files "topic"              → find relevant files
  codegraph context "task"             → full context for implementing a task
  codegraph affected "path/to/file"    → blast radius before changing a file
  codegraph index                      → full re-index after major changes
 
### FastCode CLI (when CodeGraph CLI cannot answer semantically)
  cd C:\tools\FastCode
  .\.venv\Scripts\python.exe main.py query --repo-path "FULL_PATH" --query "question"
 
### FastCode re-index after git pull (No-MCP)
FastCode stores its index as flat files (.faiss, .pkl).
After any significant git pull or major refactor:
  1. cd C:\tools\FastCode
  2. Delete: Get-ChildItem -Name "*.pkl","*.faiss" | Remove-Item
  3. Re-run query — FastCode re-indexes automatically.
'@
} else {
@'
## Integration Mode: MCP
Verify at session start with /mcp. Both servers must appear.
 
### CodeGraph MCP — session rules
The MCP server has a built-in file watcher. The graph stays current automatically.
 
Main session (lightweight — use directly):
  codegraph_search   → find symbol by name, returns file + line only
  codegraph_callers  → who calls function X
  codegraph_callees  → what does function X call
  codegraph_impact   → blast radius before any change
 
Explore subagent only (NEVER call from main session):
  codegraph_explore  → large source payload — always spawn a subagent
  codegraph_context  → large task context   — always spawn a subagent
 
When spawning Explore subagent include this instruction:
  "This project has CodeGraph (.codegraph/ exists).
   Use codegraph_explore as PRIMARY tool.
   Do NOT re-read files already returned by codegraph_explore.
   Only fall back to grep/Read for files listed under Additional relevant files."
 
### FastCode MCP — session rules and re-index
At session start:
  list_repos        → confirm repo is indexed
  list_sessions     → find prior sessions to reuse (saves tokens)
 
Semantic Q&A:
  code_qa (repos=["path"], query="question", session_id=returned_id)
  Pass session_id back for every follow-up in the same session.
 
### FastCode re-index after git pull (MCP)
FastCode has NO file watcher and NO incremental updates.
After any significant git pull or major refactor:
  1. Call remove_repo with the repo path — deletes stale .faiss/.pkl files
  2. Call code_qa — FastCode re-indexes from updated source automatically
'@
}
 
@"
# Project Operating Instructions
# Sources:
#   CodeGraph : https://github.com/colbymchenry/codegraph
#   FastCode  : https://github.com/HKUDS/FastCode
 
$integrationBlock
 
## Tool Decision Rule
 
    1. Can codegraph find this structurally?      YES → Use it. Always first.
    2. Need source code?                          YES → Explore subagent (MCP)
                                                        or codegraph context (CLI)
    3. Need semantic understanding?               YES → FastCode code_qa (MCP)
                                                        or FastCode CLI
    4. Read only files identified above.
    5. Grep only for a specific string. Last resort only.
 
Never broad-grep the whole repository.
 
## Documentation Rule
Verify all structure with CodeGraph before writing anything.
Use FastCode for narrative/conceptual sections.
Save docs to /docs/. Tutorials to /docs/TUTORIALS/.
Include exact source file and function references.
 
## Security Rule
Use CodeGraph callers/callees for source-to-sink structural tracing.
Use FastCode for exploitability reasoning after tracing.
Report only findings with verified file, function, and data path.
"@ | Set-Content -Path "CLAUDE.md" -Encoding UTF8
 
Write-Host ""
 
# ── 5. Skills ─────────────────────────────────────────────────────────────────
Write-Host "[5/5] Writing Skills..." -ForegroundColor Yellow
 
# Helper blocks used by every skill
$cgMcp = @'
### CodeGraph MCP (main session — lightweight tools only)
  codegraph_search   → find symbol, returns file + line
  codegraph_callers  → who calls X
  codegraph_callees  → what X calls
  codegraph_impact   → blast radius before change
  Explore subagent → codegraph_explore (never call from main session)
'@
 
$cgCli = @'
### CodeGraph CLI (no MCP)
  codegraph query "symbol"           → file + line
  codegraph files "topic"            → relevant files
  codegraph context "task"           → full context
  codegraph affected "path"          → blast radius
'@
 
$fcMcp = @'
### FastCode MCP
  list_repos / list_sessions         → run at session start
  code_qa (repos, query, session_id) → semantic Q&A; reuse session_id for follow-ups
  remove_repo → delete stale index. Next code_qa re-indexes from updated source.
  Use after: git pull, major refactor, significant code changes.
'@
 
$fcCli = @'
### FastCode CLI (no MCP)
  cd C:\tools\FastCode
  .\.venv\Scripts\python.exe main.py query --repo-path "PATH" --query "question"
  Re-index: Delete *.pkl and *.faiss in C:\tools\FastCode, then re-run query.
  Use after: git pull, major refactor, significant code changes.
'@
 
$cgBlock = if ($NoMCP) { $cgCli } else { $cgMcp }
$fcBlock = if ($NoMCP) { $fcCli } else { $fcMcp }
 
# ── repo-navigation ───────────────────────────────────────────────────────────
@"
---
name: repo-navigation
description: >
  Explore any repository to locate code, trace behavior, map dependencies,
  or prepare context before implementing or documenting. CodeGraph first
  for structure. FastCode for semantic questions. Never starts with grep.
---
# Repo Navigation Skill
# https://github.com/colbymchenry/codegraph | https://github.com/HKUDS/FastCode
 
$cgBlock
 
$fcBlock
 
## Execution Order
 
1. CodeGraph structural lookup (always first — zero tokens).
   Find: entry points, symbol locations, callers, callees, blast radius.
 
2. Source detail when needed.
   MCP: spawn Explore subagent with codegraph_explore instruction above.
   CLI: codegraph context "task description"
 
3. FastCode when structure is not enough (semantic, conceptual, multi-repo).
   Reuse session_id for all follow-up questions in the same session.
   After git pull: re-index first (see FastCode rules above).
 
4. Read only files identified by steps 1-3. No speculative reads.
 
5. Grep only for a specific known string. Never as exploration strategy.
 
## Output Format
1. CodeGraph findings (tools/CLI used, symbols found, files identified)
2. Explore subagent / context CLI output (if used)
3. FastCode findings (session_id, question asked, what was learned)
4. Files to read and why
5. Files excluded and why
6. Recommended next action
"@ | Set-Content -Path ".claude\skills\repo-navigation\SKILL.md" -Encoding UTF8
 
# ── documentation-writer ──────────────────────────────────────────────────────
@"
---
name: documentation-writer
description: >
  Create or update accurate project documentation grounded in verified
  repository structure. Verifies all structure with CodeGraph before
  writing. Uses FastCode for narrative/conceptual sections.
---
# Documentation Writer Skill
# https://github.com/colbymchenry/codegraph | https://github.com/HKUDS/FastCode
 
$cgBlock
 
$fcBlock
 
## Required Workflow
 
### Step 1 — Structural verification (CodeGraph)
Before writing anything:
- Find exact symbol names and file paths
- Verify described flows (callers/callees or context CLI)
- Do not document anything CodeGraph cannot confirm exists.
 
### Step 2 — Source detail
MCP: spawn Explore subagent for sections needing code.
CLI: codegraph context "doc section topic"
 
### Step 3 — Narrative synthesis (FastCode)
Use for sections explaining *why* something works, not just *what* exists.
Save session_id. Use same session for all follow-up questions about this doc.
After git pull: re-index FastCode before writing architecture docs.
 
### Step 4 — Write from verified evidence only
- File paths: from CodeGraph only
- Function/class names: exact names from CodeGraph only
- Anything unverified: mark [UNVERIFIED — needs review]
 
## Output Paths
docs/ARCHITECTURE.md · docs/GETTING_STARTED.md · docs/CORE_WORKFLOWS.md
docs/API_REFERENCE.md · docs/FEATURE_DEVELOPMENT_GUIDE.md · docs/SECURITY_MODEL.md
 
## Required Document Sections
1. Purpose — what this covers and who it is for
2. Source files — exact paths from CodeGraph
3. Main flow — verified by callers/callees
4. Key functions/classes — exact names
5. Dependencies — from CodeGraph
6. Extension points
7. Testing notes — test files from CodeGraph
8. Mermaid diagram — when flow has more than 3 steps
"@ | Set-Content -Path ".claude\skills\documentation-writer\SKILL.md" -Encoding UTF8
 
# ── tutorial-writer ───────────────────────────────────────────────────────────
@"
---
name: tutorial-writer
description: >
  Generate developer tutorials grounded in actual codebase behavior.
  Uses CodeGraph for structure, FastCode for walkthrough narrative.
  Saves to docs/TUTORIALS/.
---
# Tutorial Writer Skill
# https://github.com/colbymchenry/codegraph | https://github.com/HKUDS/FastCode
 
$cgBlock
 
$fcBlock
 
## Required Workflow
 
1. Locate the feature (CodeGraph).
   Find entry point, trace full call chain (callers + callees).
 
2. Get source detail.
   MCP: Explore subagent. CLI: codegraph context "tutorial for [feature]".
 
3. Synthesize narrative (FastCode).
   "Walk me through how [feature] works step by step for a developer
    writing a tutorial. What are the edge cases and error handling?"
   Save session_id. All follow-ups use same session.
   After git pull: re-index FastCode before writing.
 
4. Write using only verified symbol names and file paths from CodeGraph.
 
## Required Tutorial Structure
1. What You Will Learn
2. Files Involved (from CodeGraph)
3. How It Works (Mermaid diagram from callers/callees)
4. Step-by-Step Walkthrough (code from Explore subagent / context CLI)
5. Key Functions and Classes (from CodeGraph)
6. How to Extend or Modify (from codegraph_impact / affected CLI)
7. How to Test It (test files from CodeGraph)
8. Common Mistakes
 
## Output: docs/TUTORIALS/tutorial-NN-[name].md
"@ | Set-Content -Path ".claude\skills\tutorial-writer\SKILL.md" -Encoding UTF8
 
# ── security-review ───────────────────────────────────────────────────────────
@"
---
name: security-review
description: >
  Review code for security vulnerabilities using structural graph analysis
  before any file reading. Reports only findings with verified file,
  function, and traced data path.
---
# Security Review Skill
# https://github.com/colbymchenry/codegraph | https://github.com/HKUDS/FastCode
 
$cgBlock
 
$fcBlock
 
## Required Workflow
 
### Step 1 — Map attack surface (CodeGraph)
Search for security-sensitive symbols:
  authenticate, authorize, checkPermission, validateToken, session
  query, execute, raw, prepare            (database)
  readFile, writeFile, upload, unlink     (filesystem)
  parse, deserialize, fromJson, eval      (input)
  exec, spawn, system, popen             (shell)
  encrypt, decrypt, hash, secret, key    (crypto)
  outbound HTTP clients
 
For every match: trace callers (input source) and callees (sink).
Run codegraph_impact / codegraph affected for blast radius.
 
### Step 2 — Source detail
MCP: Explore subagent for suspicious paths. CLI: codegraph context.
 
### Step 3 — Exploitability reasoning (FastCode)
"Data flow: [source] → [intermediate] → [sink].
 Are there controls preventing exploitation?
 What exact input triggers this path?"
Multi-turn: save session_id, continue with follow-up questions.
After git pull: re-index FastCode before a security review of changed code.
 
## Required Finding Format
Title:      [SEVERITY] Short title
CWE/OWASP:  CWE-XXX if applicable
File:       exact/path  (from CodeGraph)
Function:   exactName() (from CodeGraph)
Data Flow:  source() → intermediate() → sink() (traced with CodeGraph)
Exploit:    exact input or state that triggers it
Controls:   what exists and why it does/does not prevent exploitation
FastCode:   semantic reasoning conclusion (if used)
Fix:        specific code change
 
Severity: CRITICAL / HIGH / MEDIUM / LOW / INFO
Unverifiable paths: [UNVERIFIED PATH — manual review required]
"@ | Set-Content -Path ".claude\skills\security-review\SKILL.md" -Encoding UTF8
 
# ── feature-planner ───────────────────────────────────────────────────────────
@"
---
name: feature-planner
description: >
  Plan a new feature by finding the analogous pattern with CodeGraph,
  assessing blast radius, synthesizing approach with FastCode, and
  producing a plan before any code is written. Never writes code before
  plan is approved.
---
# Feature Planner Skill
# https://github.com/colbymchenry/codegraph | https://github.com/HKUDS/FastCode
 
$cgBlock
 
$fcBlock
 
## Core Rule: Never write code before the plan is approved.
 
## Required Workflow
 
### Step 1 — Find analogous pattern (CodeGraph)
  codegraph_search / query → find similar existing feature
  codegraph_callers / callees → integration points and dependencies
  codegraph_impact / affected → blast radius of analogous feature
 
### Step 2 — Get implementation detail
MCP: Explore subagent → "Show full implementation of [analogous feature]."
CLI: codegraph context "[analogous feature] full implementation"
 
### Step 3 — Synthesize approach (FastCode)
"I want to add [FEATURE]. I found [ANALOGOUS FEATURE] as pattern.
 Best approach? Architectural constraints? Risks?"
Follow-up (same session_id):
"What tests? What edge cases does [ANALOGOUS FEATURE] handle to replicate?"
After git pull affecting the analogous area: re-index FastCode first.
 
### Step 4 — Produce the plan
 
Feature: [Name]
Summary: [One paragraph]
Analogous Feature: file (CodeGraph), symbol (CodeGraph), why analogous
Files to Create: path — purpose
Files to Modify: path — what changes (verified by blast radius)
Files NOT to Touch: [prevents scope creep]
Implementation Steps: [with exact file + function from CodeGraph]
Tests Required: [test files from CodeGraph]
Risks: [from blast radius + FastCode]
 
### Step 5 — Wait for approval
Do not write any code until user approves.
Revise and re-present if changes are requested.
"@ | Set-Content -Path ".claude\skills\feature-planner\SKILL.md" -Encoding UTF8
 
Write-Host "   Skills written." -ForegroundColor Green
Write-Host ""
 
# ── Summary ───────────────────────────────────────────────────────────────────
Write-Host "=== Done ===" -ForegroundColor Green
Write-Host ""
Write-Host "  Project : $ProjectPath"
Write-Host "  Mode    : $(if ($NoMCP) { 'No-MCP' } else { 'MCP' })"
Write-Host "  Created : CLAUDE.md, .claude/settings.json, 5 skills, docs/"
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
if ($NoMCP) {
    Write-Host "  1. Customize CLAUDE.md for this project."
    Write-Host "  2. claude"
    Write-Host "  3. Test: Use Bash to run 'codegraph query ""main""'"
    Write-Host "  4. /skills  (confirm 5 skills)"
} else {
    Write-Host "  1. Customize CLAUDE.md for this project."
    Write-Host "  2. claude"
    Write-Host "  3. /mcp  (codegraph + fastcode must appear)"
    Write-Host "  4. /skills  (5 skills must appear)"
    Write-Host "  5. Test: Use codegraph_search to find the main entry point."
}
Write-Host ""
Write-Host "After git pull or major refactor:" -ForegroundColor Yellow
if ($NoMCP) {
    Write-Host "  CodeGraph : codegraph index  (or auto-synced per-edit by hook)"
    Write-Host "  FastCode  : delete *.pkl/*.faiss in C:\tools\FastCode, then re-run query"
} else {
    Write-Host "  CodeGraph : auto-handled by built-in file watcher — nothing to do"
    Write-Host "  FastCode  : call remove_repo, then code_qa to force re-index"
}
Write-Host ""
Write-Host "Update Claude Code: winget upgrade Anthropic.ClaudeCode" -ForegroundColor Gray
```
 
**Usage:**
```powershell
# MCP mode (default)
.\setup-claude-project.ps1 "C:\repo\my-project"
 
# No-MCP mode (enterprise / MCP blocked)
.\setup-claude-project.ps1 "C:\repo\my-project" -NoMCP
```
 
---
 
## Section 7 — Verify After Bootstrap
 
### MCP
```
claude
/mcp      ← codegraph + fastcode must appear
/skills   ← repo-navigation, documentation-writer, tutorial-writer,
             security-review, feature-planner must appear
```
 
Test:
```
Use codegraph_search to find the main entry point.
Then spawn an Explore agent to explain what it does.
Do NOT call codegraph_explore from this main session.
```
 
### No-MCP
```powershell
codegraph query "main"     # must return results
codegraph status           # Backend: native
```
Inside Claude Code:
```
/skills    ← 5 skills must appear
Use Bash to run: codegraph files "authentication"
```
 
---
 
## Section 8 — Daily Workflows
 
### Start of session (MCP)
```
/mcp               ← verify both servers
list_repos         ← confirm FastCode has this repo indexed
list_sessions      ← find prior sessions to reuse (saves tokens)
```
 
### After git pull or major refactor
```
CodeGraph MCP  : nothing — watcher handled it
CodeGraph CLI  : codegraph index
FastCode MCP   : remove_repo → then code_qa (re-indexes automatically)
FastCode CLI   : delete *.pkl/*.faiss in C:\tools\FastCode → re-run query
```
 
### Feature work
```
/feature-planner I need to implement [FEATURE].
Find the analogous pattern with CodeGraph first.
Produce a plan and wait for my approval before writing code.
```
 
### Update Claude Code
```powershell
winget upgrade Anthropic.ClaudeCode
```
 
---
 
## Section 9 — Troubleshooting
 
| Problem | Fix |
|---|---|
| `claude` not found | Add `%USERPROFILE%\.local\bin` to PATH; reopen PowerShell |
| `codegraph status` shows `Backend: wasm` | Install VS Build Tools; `npm rebuild better-sqlite3` |
| CodeGraph tools not in `/mcp` | `codegraph init -i` in project root; restart Claude Code; `/doctor` |
| FastCode not connecting (MCP) | Test `python mcp_server.py` manually; check `.env`; verify JSON uses `\\` |
| FastCode answers look stale | Re-index: `remove_repo` then `code_qa` (MCP) or delete `.pkl`/`.faiss` (CLI) |
| `codegraph init` says already initialized | Run `codegraph index` instead |
 
---
 
## Quick Reference
 
```
SYNC BEHAVIOUR (sourced from both GitHub READMEs):
 
  CodeGraph MCP  → built-in file watcher, auto-syncs every 2 sec
                   Source: https://github.com/colbymchenry/codegraph
  CodeGraph CLI  → no watcher; PostToolUse hook calls codegraph sync
  FastCode       → NO watcher, NO incremental sync (either mode)
                   After git pull: remove_repo+code_qa (MCP) or delete .pkl/.faiss (CLI)
                   Source: https://github.com/HKUDS/FastCode
 
HOOKS (only one is justified):
  PostToolUse → codegraph sync   No-MCP only — replaces absent watcher
  Nothing for CodeGraph MCP      watcher already running
  Nothing for FastCode           no incremental sync exists
 
CODEGRAPH SESSION RULES:
  Main session:       codegraph_search, callers, callees, impact
  Explore agent only: codegraph_explore, codegraph_context
 
FASTCODE MCP TOOLS:
  list_repos / list_sessions     → session start, every time
  code_qa (repos, query, session_id) → semantic Q&A
  remove_repo                    → clear stale index after git pull
 
FASTCODE CLI:
  python main.py query --repo-path "PATH" --query "QUESTION"
  Re-index: delete *.pkl/*.faiss in C:\tools\FastCode
 
DECISION ORDER:
  1. codegraph_search / callers / callees / impact  (free, instant)
  2. Explore subagent with codegraph_explore        (source code needed)
  3. FastCode code_qa or CLI                        (meaning needed)
  4. Read specific identified files
  5. Grep — last resort, specific string only
  NEVER: broad grep as first step
 
Sources:
  CodeGraph  : https://github.com/colbymchenry/codegraph
  FastCode   : https://github.com/HKUDS/FastCode
  Hooks docs : https://code.claude.com/docs/en/hooks-guide
  Skills docs: https://code.claude.com/docs/en/skills
  Setup docs : https://code.claude.com/docs/en/setup
```
 
