Setup claude code windows · MD
# Claude Code — Windows Setup
## FastCode + CodeGraph + Skills + Hooks (MCP and No-MCP paths)
 
> **How to use:** Save as `SETUP_CLAUDE_CODE_WINDOWS.md` anywhere.
> Start Claude Code with `claude`, then say:
> "Read SETUP_CLAUDE_CODE_WINDOWS.md and execute every phase in order.
>  Stop and ask me if any step fails before continuing."
 
---
 
## Section 0 — Mental Model
 
```
FastCode  = WHAT does this code mean?
            Semantic Q&A, architecture summaries, cross-file understanding.
            Requires LLM API key. Costs tokens. Use selectively.
            Source: https://github.com/HKUDS/FastCode
 
CodeGraph = WHERE is everything? HOW is it connected?
            AST graph, SQLite, zero tokens, 100% local, always first.
            Source: https://github.com/colbymchenry/codegraph
 
Claude    = Write it. Explain it. Document it. Test it.
```
 
### CodeGraph session split rule — from the official README
 
NEVER call `codegraph_explore` or `codegraph_context` from the main
Claude Code session. They return large source payloads that fill context.
 
```
Main session  (lightweight, use directly):
  codegraph_search    → find a symbol by name → file + line only, no code
  codegraph_callers   → who calls function X
  codegraph_callees   → what does function X call
  codegraph_impact    → full blast radius before any change
 
Explore subagent only (never call from main session):
  codegraph_explore   → full source sections, use inside a spawned subagent
  codegraph_context   → full task context, use inside a spawned subagent
```
 
### Tool decision rule
 
```
1. Can codegraph_search / callers / callees / impact answer this?
      YES → Use it. Zero tokens. Always first.
       NO → Need source code?
              YES → Spawn Explore subagent → use codegraph_explore inside it
               NO → Need semantic understanding?
                     MCP available (/mcp shows fastcode) → code_qa
                     MCP blocked → Bash: python main.py query --repo-path ...
                     Neither available → Read only files identified above
```
 
Never start with broad grep across the whole repository.
 
### Frequency table
 
| Action | Frequency |
|---|---|
| `npm install -g @colbymchenry/codegraph` | Once ever on this machine |
| `npx @colbymchenry/codegraph` (global installer) | Once ever — writes MCP config |
| `codegraph init -i` | Once per project |
| `codegraph index` | After major code changes |
| `codegraph sync` | Automatic — file watcher handles it |
| FastCode clone + venv + deps | Once ever on this machine |
| `winget upgrade Anthropic.ClaudeCode` | Periodically — no auto-update |
 
---
 
## Section 1 — Whether MCP Is Available
 
Before installing anything, determine your situation. This decides
which integration path you follow for the rest of this guide.
 
Open PowerShell and run:
```powershell
claude
```
Then inside Claude Code:
```
/mcp
```
 
**Situation A — MCP works** (codegraph and fastcode appear after setup):
Follow the MCP path in every section below. This is the full integration.
 
**Situation B — MCP is blocked by enterprise policy**
(admin has set `allowManagedHooksOnly` in managed settings,
or `/mcp` shows no servers load):
Follow the No-MCP path. Both CodeGraph and FastCode expose CLI interfaces.
Claude calls them via the `Bash` tool. Hooks handle auto-sync.
You lose the seamless MCP protocol but keep full functionality.
 
**Situation C — MCP misconfigured** (servers listed but tools missing):
Run `/doctor` inside Claude Code.
Check that `codegraph init -i` was run inside the project directory.
Restart Claude Code and retry.
Source: https://code.claude.com/docs/en/hooks-guide
 
---
 
## Section 2 — Prerequisites
 
```powershell
claude --version
git --version
node --version
npm --version
python --version
winget --version
```
 
Install anything missing. Close and reopen PowerShell after each install:
 
```powershell
# Claude Code (if missing)
irm https://claude.ai/install.ps1 | iex
# or: winget install --id Anthropic.ClaudeCode -e --source winget
 
# Git (required by Claude Code internally — Git Bash is used on Windows)
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
# Close and reopen PowerShell, then retry: claude --version
```
 
---
 
## Section 3 — Install CodeGraph (once ever on this machine)
 
Source: https://github.com/colbymchenry/codegraph
 
### Step 3a — Install globally
 
```powershell
npm install -g @colbymchenry/codegraph
codegraph --version
```
 
### Step 3b — Run the global interactive installer
 
This writes the MCP server config to `~/.claude.json`, sets auto-allow
permissions, and adds global instructions to `~/.claude/CLAUDE.md`.
Run it once. Never per project.
 
```powershell
npx @colbymchenry/codegraph
```
 
When prompted: select **Claude Code**, then **global**.
 
Non-interactive (for scripting):
```powershell
codegraph install --target=claude --location=global --yes
```
 
### Step 3c — Fix Windows backend if needed
 
```powershell
codegraph status
```
 
Must show `Backend: native`. If it shows `Backend: wasm` (5–10x slower):
```powershell
winget install --id Microsoft.VisualStudio.2022.BuildTools -e --source winget
# Close and reopen PowerShell, then:
npm rebuild better-sqlite3
codegraph status   # must now show Backend: native
```
 
### Step 3d — Verify MCP registration
 
```powershell
claude
```
```
/mcp
```
`codegraph` must appear. If not: run `/doctor`.
 
**Manual MCP config** (if the installer did not write it automatically):
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
 
---
 
## Section 4 — Install FastCode (once ever on this machine)
 
Source: https://github.com/HKUDS/FastCode
 
FastCode is optional. All skills degrade gracefully to CodeGraph-only
or CLI-only mode when FastCode is unavailable.
 
```powershell
# Install uv (fast Python package manager)
pip install uv
uv --version
 
# Clone to a permanent tools location outside any project
New-Item -ItemType Directory -Force -Path "C:\tools"
cd C:\tools
git clone https://github.com/HKUDS/FastCode.git
cd FastCode
 
# Read the README before installing — follow it if it differs from below
# Current steps match v1.0.1
uv venv --python=3.12
.\.venv\Scripts\Activate.ps1    # Windows path — NOT source .venv/bin/activate
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
# Any OpenAI-compatible provider works
OPENAI_API_KEY=sk-your-key-here
MODEL=gpt-4o
BASE_URL=https://api.openai.com/v1
 
# For Anthropic Claude API:
# OPENAI_API_KEY=sk-ant-your-key-here
# MODEL=claude-sonnet-4-6
# BASE_URL=https://api.anthropic.com/v1
```
 
**Test the MCP server before registering:**
```powershell
cd C:\tools\FastCode
.\.venv\Scripts\Activate.ps1
python mcp_server.py    # must start without errors; Ctrl+C to stop
deactivate
```
 
**Register with Claude Code (MCP path):**
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
 
**Verify both servers:**
```
/mcp    ← codegraph and fastcode must both appear
```
 
---
 
## Section 5 — No-MCP Integration via Bash CLI and Hooks
 
**Skip this section if both MCP servers appear in `/mcp`.**
 
When MCP is blocked by enterprise policy, Claude calls CodeGraph and
FastCode as command-line tools via the `Bash` tool. Hooks handle
auto-sync. No MCP protocol is involved.
 
Source: https://code.claude.com/docs/en/hooks-guide
 
### CodeGraph CLI commands (Bash tool, no MCP)
 
These subcommands are documented in the CodeGraph CLAUDE.md
(https://github.com/colbymchenry/codegraph/blob/main/CLAUDE.md):
 
```powershell
# Find a symbol — returns file + line
codegraph query "UserService"
 
# Find relevant files for a topic
codegraph files "authentication"
 
# Build task context
codegraph context "add retry logic to payment processor"
 
# Show blast radius of a file
codegraph affected "src/auth/session.ts"
 
# Check status
codegraph status
 
# Re-index after major changes
codegraph index
```
 
Claude calls these inside the `Bash` tool. Example in CLAUDE.md:
```
Run: codegraph query "login" to find the entry point before reading files.
```
 
### FastCode CLI command (Bash tool, no MCP)
 
From the FastCode README (https://github.com/HKUDS/FastCode):
```powershell
# Activate venv first, then query
cd C:\tools\FastCode
.\.venv\Scripts\Activate.ps1
python main.py query --repo-path "C:\path\to\project" --query "How does auth work?"
```
 
Claude calls this via Bash when FastCode MCP is unavailable.
 
### Hooks for auto-sync (no MCP required)
 
Hooks are defined in `.claude/settings.json` inside the project.
They fire deterministically — unlike CLAUDE.md guidelines they always run.
Source: https://code.claude.com/docs/en/hooks-guide
 
**Auto-sync CodeGraph after every file edit:**
```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "codegraph sync"
          }
        ]
      }
    ]
  }
}
```
 
**Re-inject CodeGraph context after compaction** (context window reset):
```json
{
  "hooks": {
    "SessionStart": [
      {
        "matcher": "compact",
        "hooks": [
          {
            "type": "command",
            "command": "echo 'CodeGraph is initialized. Use Bash to run: codegraph query, codegraph files, codegraph context, codegraph affected. FastCode CLI: python C:\\tools\\FastCode\\main.py query --repo-path \"%CLAUDE_PROJECT_DIR%\" --query \"your question\"'"
          }
        ]
      }
    ]
  }
}
```
 
**No-MCP CLAUDE.md instructions** (add this block when MCP is blocked):
```markdown
## No-MCP Mode — CodeGraph via Bash
 
MCP is not available. Use the Bash tool to call CodeGraph CLI directly.
 
Bash commands for CodeGraph (run from project root):
  codegraph query "symbol"               → find a symbol, returns file + line
  codegraph files "topic"                → find relevant files for a topic
  codegraph context "task description"   → get full context for a task
  codegraph affected "path/to/file"      → blast radius of a file change
  codegraph index                        → re-index after major changes
 
Rule: always run codegraph query or codegraph files BEFORE reading any file.
Never grep the whole repo as a first step.
 
## No-MCP Mode — FastCode via Bash
 
Bash command for FastCode semantic Q&A (when codegraph CLI cannot answer):
  cd C:\tools\FastCode
  .\.venv\Scripts\Activate.ps1
  python main.py query --repo-path "FULL_PATH_TO_PROJECT" --query "your question"
 
Use FastCode when:
  - codegraph CLI returns no result after retry
  - The question needs semantic understanding (why/how), not just structure
```
 
---
 
## Section 6 — Per-Project Bootstrap Script
 
Save as `C:\tools\setup-claude-project.ps1`.
Run once per new project. Never for already-initialized projects.
 
```powershell
# setup-claude-project.ps1
# Usage: .\setup-claude-project.ps1 "C:\path\to\project" [-NoMCP]
#
# Sources:
#   CodeGraph : https://github.com/colbymchenry/codegraph
#   FastCode  : https://github.com/HKUDS/FastCode
#   Hooks     : https://code.claude.com/docs/en/hooks-guide
 
param(
    [Parameter(Mandatory=$true)][string]$ProjectPath,
    [switch]$NoMCP   # pass -NoMCP when MCP is blocked by enterprise policy
)
 
$ProjectPath = Resolve-Path $ProjectPath -ErrorAction Stop
Write-Host ""
Write-Host "=== Claude Code Project Bootstrap ===" -ForegroundColor Cyan
Write-Host "Project : $ProjectPath" -ForegroundColor Cyan
Write-Host "Mode    : $(if ($NoMCP) { 'No-MCP (Bash CLI + Hooks)' } else { 'MCP' })" -ForegroundColor Cyan
Write-Host ""
 
Set-Location $ProjectPath
 
# ── Step 1: CodeGraph per-project init ───────────────────────────────────────
# codegraph init -i initializes AND indexes in one step.
# Source: https://github.com/colbymchenry/codegraph README Quick Start
Write-Host "[1/6] CodeGraph init..." -ForegroundColor Yellow
 
if (Test-Path ".codegraph") {
    Write-Host "      .codegraph/ already exists — running codegraph index to refresh." -ForegroundColor Gray
    codegraph index
} else {
    codegraph init -i
}
 
if ($LASTEXITCODE -ne 0) {
    Write-Error "codegraph init failed. Run 'codegraph status' and 'codegraph --help'."
    exit 1
}
codegraph status
Write-Host ""
 
# ── Step 2: Directory structure ───────────────────────────────────────────────
Write-Host "[2/6] Creating project structure..." -ForegroundColor Yellow
 
New-Item -ItemType Directory -Force -Path ".claude"                              | Out-Null
New-Item -ItemType Directory -Force -Path ".claude\skills"                       | Out-Null
New-Item -ItemType Directory -Force -Path ".claude\skills\repo-navigation"       | Out-Null
New-Item -ItemType Directory -Force -Path ".claude\skills\documentation-writer"  | Out-Null
New-Item -ItemType Directory -Force -Path ".claude\skills\tutorial-writer"       | Out-Null
New-Item -ItemType Directory -Force -Path ".claude\skills\security-review"       | Out-Null
New-Item -ItemType Directory -Force -Path ".claude\skills\feature-planner"       | Out-Null
New-Item -ItemType Directory -Force -Path ".claude\hooks"                        | Out-Null
New-Item -ItemType Directory -Force -Path "docs"                                 | Out-Null
New-Item -ItemType Directory -Force -Path "docs\TUTORIALS"                       | Out-Null
Write-Host ""
 
# ── Step 3: .claude/settings.json — hooks ────────────────────────────────────
Write-Host "[3/6] Writing .claude/settings.json (hooks)..." -ForegroundColor Yellow
 
$settingsJson = if ($NoMCP) {
    # No-MCP: hooks do auto-sync + context re-injection after compaction
    @'
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "codegraph sync"
          }
        ]
      }
    ],
    "SessionStart": [
      {
        "matcher": "compact",
        "hooks": [
          {
            "type": "command",
            "command": "echo 'No-MCP mode. CodeGraph CLI: codegraph query \"symbol\", codegraph files \"topic\", codegraph context \"task\", codegraph affected \"file\". FastCode CLI: cd C:\\tools\\FastCode && .venv\\Scripts\\python.exe main.py query --repo-path \"%CLAUDE_PROJECT_DIR%\" --query \"question\"'"
          }
        ]
      }
    ]
  }
}
'@
} else {
    # MCP mode: only need auto-sync hook; codegraph MCP server handles the rest
    @'
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "codegraph sync"
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
 
# ── Step 4: CLAUDE.md ─────────────────────────────────────────────────────────
Write-Host "[4/6] Writing CLAUDE.md..." -ForegroundColor Yellow
 
$mcpBlock = if ($NoMCP) {
@'
 
## Integration Mode: No-MCP (Bash CLI)
 
MCP is not available on this machine. Use the Bash tool to call tools directly.
 
### CodeGraph CLI (always first — zero tokens)
Run these from the project root before reading any file:
  codegraph query "symbol"             → find a symbol, returns file + line
  codegraph files "topic"              → find relevant files
  codegraph context "task description" → get full context for a task
  codegraph affected "path/to/file"    → blast radius before changing a file
  codegraph index                      → re-index after major code changes
 
### FastCode CLI (when codegraph CLI cannot answer semantically)
  cd C:\tools\FastCode
  .\.venv\Scripts\python.exe main.py query --repo-path "FULL_PATH" --query "question"
 
Use FastCode CLI when:
  - codegraph CLI returns nothing after retry with shorter term
  - The question needs conceptual understanding, not structural lookup
  - The question spans multiple repos (run once per repo, compare answers)
 
Never grep the whole repo. Never read files without running codegraph CLI first.
'@
} else {
@'
 
## Integration Mode: MCP
 
Both CodeGraph and FastCode run as MCP servers. Verify with /mcp at session start.
 
### CodeGraph MCP — main session tools (lightweight, use directly)
  codegraph_search   → find a symbol by name → file + line only
  codegraph_callers  → who calls function X
  codegraph_callees  → what does function X call
  codegraph_impact   → blast radius before any change
 
### CodeGraph MCP — Explore subagent only (NEVER in main session)
  codegraph_explore  → full source sections — always spawn a subagent for this
  codegraph_context  → full task context   — always spawn a subagent for this
 
When spawning an Explore subagent include:
  "This project has CodeGraph (.codegraph/ exists).
   Use codegraph_explore as PRIMARY tool.
   Do NOT re-read files already returned by codegraph_explore.
   Only fall back to grep/Read for files listed under Additional relevant files."
 
### FastCode MCP tools (check /mcp first)
  list_repos        → what is already indexed (run at session start)
  list_sessions     → prior sessions to reuse (run at session start)
  code_qa           → semantic Q&A
                      repos=["/absolute/path"], query="question"
                      session_id=returned_id for multi-turn follow-ups
  get_session       → review prior analysis
  remove_repo       → force fresh re-index after major refactor
  delete_session    → clean up stale sessions
 
Use code_qa when:
  - codegraph_search returns nothing after retry with shorter term
  - The question needs conceptual understanding, not structural lookup
  - Question spans multiple repos → pass both paths in repos=[]
  - New developer needs architecture overview
'@
}
 
$claudeMd = @"
# Project Operating Instructions
# Sources:
#   CodeGraph : https://github.com/colbymchenry/codegraph
#   FastCode  : https://github.com/HKUDS/FastCode
#   Hooks     : https://code.claude.com/docs/en/hooks-guide
$mcpBlock
 
## Tool Decision Rule
 
```
1. Can codegraph find this structurally?    YES → Use it. Always first.
2. Need source code?                        YES → Explore subagent (MCP)
                                                   or codegraph context (CLI)
3. Need semantic understanding?             YES → FastCode code_qa (MCP)
                                                   or FastCode CLI
4. Read only identified files.
5. Grep only as last resort for a specific string.
Never broad-grep the whole repo.
```
 
## Documentation Rule
Verify all structure with CodeGraph before writing.
Use FastCode for narrative/conceptual sections.
Save docs to /docs/. Save tutorials to /docs/TUTORIALS/.
Include exact source file and function references.
 
## Security Rule
Use CodeGraph callers/callees for source-to-sink tracing.
Use FastCode for exploitability reasoning after tracing.
Report only findings with verified file, function, and data path.
 
## Reindexing Rule
Run codegraph index after major code changes.
If using FastCode MCP: call remove_repo then code_qa to force fresh semantic index.
If using FastCode CLI: re-run python main.py query — it re-indexes automatically.
"@
 
Set-Content -Path "CLAUDE.md" -Value $claudeMd -Encoding UTF8
Write-Host ""
 
# ── Step 5: Skills ────────────────────────────────────────────────────────────
Write-Host "[5/6] Writing Skills..." -ForegroundColor Yellow
 
# Shared helper — produces the correct "how to use CodeGraph" block per mode
function Get-CodeGraphBlock($mode) {
    if ($mode -eq "nomcp") {
        return @'
### CodeGraph — Bash CLI (no MCP)
Run from project root before reading any file:
  codegraph query "symbol"             → file + line, no code
  codegraph files "topic"              → relevant file list
  codegraph context "task"             → full context for a task
  codegraph affected "path/to/file"    → blast radius
'@
    } else {
        return @'
### CodeGraph — MCP tools
Main session only (lightweight):
  codegraph_search   → symbol location, no code
  codegraph_callers  → who calls X
  codegraph_callees  → what X calls
  codegraph_impact   → blast radius
 
Source code (Explore subagent only — never call from main session):
  Spawn subagent: "Use codegraph_explore. Do not re-read files it returns."
'@
    }
}
 
function Get-FastCodeBlock($mode) {
    if ($mode -eq "nomcp") {
        return @'
### FastCode — Bash CLI (no MCP)
Use when codegraph CLI cannot answer semantically:
  cd C:\tools\FastCode
  .\.venv\Scripts\python.exe main.py query `
    --repo-path "FULL_PATH_TO_PROJECT" `
    --query "your question here"
'@
    } else {
        return @'
### FastCode — MCP tools
Check /mcp before using. If fastcode not listed, fall back to CLI or skip.
  list_repos / list_sessions  → run at session start
  code_qa (repos, query, session_id) → semantic Q&A; pass session_id for follow-ups
  remove_repo → force re-index after major refactor
'@
    }
}
 
$codeMode = if ($NoMCP) { "nomcp" } else { "mcp" }
$cgBlock = Get-CodeGraphBlock $codeMode
$fcBlock = Get-FastCodeBlock $codeMode
 
# ── repo-navigation ──────────────────────────────────────────────────────────
Set-Content -Path ".claude\skills\repo-navigation\SKILL.md" -Encoding UTF8 -Value @"
---
name: repo-navigation
description: >
  Explore any repository to locate code, trace behavior, map dependencies,
  or prepare context before implementing or documenting. Applies CodeGraph
  first, FastCode for semantic questions. Never starts with broad grep.
  Works in both MCP mode and no-MCP Bash CLI mode.
---
 
# Repo Navigation Skill
# Sources: https://github.com/colbymchenry/codegraph
#          https://github.com/HKUDS/FastCode
 
$cgBlock
 
$fcBlock
 
## Execution Order
 
1. CodeGraph structural lookup (always first, zero tokens).
   Use lightweight tools (MCP) or CLI commands (no-MCP).
   Find: entry points, symbol locations, callers, callees, blast radius.
 
2. Source detail when needed.
   MCP: spawn Explore subagent with codegraph_explore.
   No-MCP: run codegraph context "task description" via Bash.
 
3. FastCode semantic question (when structure is not enough).
   MCP: code_qa with repos and query. Reuse session_id for follow-ups.
   No-MCP: python main.py query --repo-path ... --query ...
 
4. Read only files identified by steps 1-3. No speculative reads.
 
5. Grep only for a specific known string. Never as exploration.
 
## If Both Tools Unavailable
Report: "CodeGraph and FastCode unavailable. Run /mcp to check MCP status."
Ask permission before falling back to directory browsing.
 
## Output Format
1. CodeGraph findings (tools or CLI commands used, what was returned)
2. Explore subagent / codegraph context output (if used)
3. FastCode findings (MCP or CLI, question asked, session_id if MCP)
4. Files identified as relevant — source: CodeGraph / FastCode / inference
5. Files excluded and why
6. Recommended next action
"@
 
# ── documentation-writer ─────────────────────────────────────────────────────
Set-Content -Path ".claude\skills\documentation-writer\SKILL.md" -Encoding UTF8 -Value @"
---
name: documentation-writer
description: >
  Create or update accurate project documentation grounded in verified
  repository structure. Use when asked to write architecture docs, API
  references, workflow guides, or getting-started documentation.
  Works in both MCP mode and no-MCP Bash CLI mode.
---
 
# Documentation Writer Skill
# Sources: https://github.com/colbymchenry/codegraph
#          https://github.com/HKUDS/FastCode
 
$cgBlock
 
$fcBlock
 
## Required Workflow
 
### Step 1 — Verify structure (CodeGraph)
Before writing a single sentence:
- Find exact symbol names and file paths (search / query CLI)
- Verify flows (callers / callees or context CLI)
- Do not document anything CodeGraph cannot confirm exists.
 
### Step 2 — Source detail (Explore subagent or codegraph context CLI)
MCP: spawn Explore subagent with codegraph_explore for sections needing code.
No-MCP: run codegraph context "topic of this doc section" via Bash.
 
### Step 3 — Narrative synthesis (FastCode)
Use FastCode for sections requiring explanation of *why* something works:
- MCP: code_qa "Explain [module] for a developer reading the docs"
       Save session_id. Use it for all follow-up questions about the same doc.
- No-MCP: python main.py query --repo-path ... --query "Explain [module]..."
 
### Step 4 — Write from verified evidence only
- Every file path: from CodeGraph
- Every function/class name: exact names from CodeGraph
- Every flow description: verified by callers/callees or context CLI
- Anything unverified: mark [UNVERIFIED — needs review]
 
## Output Paths
docs/ARCHITECTURE.md · docs/GETTING_STARTED.md · docs/CORE_WORKFLOWS.md
docs/API_REFERENCE.md · docs/FEATURE_DEVELOPMENT_GUIDE.md · docs/SECURITY_MODEL.md
 
## Required Document Sections
1. Purpose — what this covers and who it is for
2. Relevant source files — exact paths from CodeGraph
3. Main flow — entry to output, verified by callers/callees
4. Key functions/classes — exact names, what they do
5. Dependencies — verified by CodeGraph
6. Extension points — where to add new behavior
7. Testing notes — test files from CodeGraph
8. Mermaid diagram — when flow has more than 3 steps
"@
 
# ── tutorial-writer ──────────────────────────────────────────────────────────
Set-Content -Path ".claude\skills\tutorial-writer\SKILL.md" -Encoding UTF8 -Value @"
---
name: tutorial-writer
description: >
  Generate developer tutorials grounded in actual codebase behavior.
  Use when asked to write a step-by-step tutorial, walkthrough, or
  how-to guide for a feature or pattern. Saves to docs/TUTORIALS/.
  Works in both MCP mode and no-MCP Bash CLI mode.
---
 
# Tutorial Writer Skill
# Sources: https://github.com/colbymchenry/codegraph
#          https://github.com/HKUDS/FastCode
 
$cgBlock
 
$fcBlock
 
## Required Workflow
 
### Step 1 — Locate the feature (CodeGraph)
- Find entry point by symbol name (search / query CLI)
- Trace the call chain (callers + callees or context CLI)
- Confirm: entry point, service layer, data layer, tests
 
### Step 2 — Get source detail
MCP: spawn Explore subagent — "Show implementation of [feature]
     from entry point to output."
No-MCP: codegraph context "tutorial for [feature]" via Bash.
 
### Step 3 — Synthesize the narrative (FastCode)
MCP: code_qa "Walk me through how [feature] works step by step
     for a developer writing a tutorial."
     Save session_id. Follow up: "What are edge cases and error handling?"
No-MCP: python main.py query --repo-path ... --query "Walk me through..."
 
### Step 4 — Write tutorial using only verified names
 
## Required Tutorial Structure
1. What You Will Learn
2. Files Involved (exact paths from CodeGraph)
3. How It Works (Mermaid diagram from callers/callees)
4. Step-by-Step Walkthrough (code from Explore subagent or context CLI)
5. Key Functions and Classes (from CodeGraph)
6. How to Extend or Modify (from blast radius / codegraph_impact or affected CLI)
7. How to Test It (test files from CodeGraph)
8. Common Mistakes
 
## Output Location
docs/TUTORIALS/tutorial-NN-[short-name].md
Increment NN from the highest existing number.
"@
 
# ── security-review ──────────────────────────────────────────────────────────
Set-Content -Path ".claude\skills\security-review\SKILL.md" -Encoding UTF8 -Value @"
---
name: security-review
description: >
  Review code for security vulnerabilities using structural graph analysis
  before any file reading. Reports only evidence-backed findings with exact
  file, function, and traced data path. Works in MCP and no-MCP mode.
---
 
# Security Review Skill
# Sources: https://github.com/colbymchenry/codegraph
#          https://github.com/HKUDS/FastCode
 
$cgBlock
 
$fcBlock
 
## Required Workflow
 
### Step 1 — Map attack surface (CodeGraph, no code yet)
Search for security-sensitive symbols:
  authenticate, authorize, checkPermission, validateToken, session
  query, execute, raw, prepare  (database)
  readFile, writeFile, upload, unlink  (filesystem)
  parse, deserialize, fromJson, eval  (input)
  exec, spawn, system, popen  (shell)
  encrypt, decrypt, hash, secret, key  (crypto)
  outbound HTTP client calls
 
For every match: trace callers (input path) and callees (sink path).
MCP: codegraph_callers, codegraph_callees, codegraph_impact
No-MCP: codegraph context "security audit of [function]" via Bash
 
### Step 2 — Source detail (Explore subagent or context CLI)
MCP: spawn Explore subagent for suspicious paths found in Step 1.
No-MCP: codegraph context "source-to-sink path for [function]" via Bash.
 
### Step 3 — Exploitability reasoning (FastCode)
MCP: code_qa "Data flow: [source] → [intermediate] → [sink].
     Are there controls preventing exploitation? What input triggers this?"
     Multi-turn: save session_id, continue with follow-up questions.
No-MCP: python main.py query --repo-path ... --query "same question"
 
### Step 4 — Write findings (evidence only)
 
## Required Finding Format
Title:       [SEVERITY] Short descriptive title
CWE/OWASP:   CWE-XXX if applicable
File:        exact/path/file.ext  (from CodeGraph)
Function:    exactFunctionName()  (from CodeGraph)
Data Flow:   source() → intermediate() → sink()  (traced with CodeGraph)
Exploit:     exact input or state that triggers it
Controls:    what exists and why it does/does not prevent exploitation
FastCode:    semantic reasoning conclusion (if used)
Fix:         specific code change recommendation
 
## Severity: CRITICAL / HIGH / MEDIUM / LOW / INFO
 
## Rules
Every finding requires file + function from CodeGraph.
Unverifiable paths: mark [UNVERIFIED PATH — manual review required].
Missing controls: state explicitly ("no auth middleware on this route").
"@
 
# ── feature-planner ──────────────────────────────────────────────────────────
Set-Content -Path ".claude\skills\feature-planner\SKILL.md" -Encoding UTF8 -Value @"
---
name: feature-planner
description: >
  Plan a new feature by finding the analogous pattern with CodeGraph,
  assessing blast radius, synthesizing with FastCode, and producing a
  concrete plan before any code is written. Never writes code before
  plan is approved. Works in MCP and no-MCP mode.
---
 
# Feature Planner Skill
# Sources: https://github.com/colbymchenry/codegraph
#          https://github.com/HKUDS/FastCode
 
$cgBlock
 
$fcBlock
 
## Core Rule
Never write code before the plan is approved by the user.
 
## Required Workflow
 
### Step 1 — Find analogous pattern (CodeGraph)
MCP: codegraph_search → find similar existing feature by symbol name
     codegraph_callers / codegraph_callees → integration points
     codegraph_impact → blast radius of analogous feature
No-MCP: codegraph query "similar feature name"
        codegraph context "how does [analogous feature] work"
        codegraph affected "path/to/analogous/file"
 
### Step 2 — Get implementation detail
MCP: spawn Explore subagent — "Show full implementation of [analogous feature]."
No-MCP: codegraph context "[analogous feature] implementation" via Bash.
 
### Step 3 — Synthesize approach (FastCode)
MCP: code_qa "I want to add [FEATURE]. I found [ANALOGOUS FEATURE] as pattern.
     Best approach? Architectural constraints? Risks?"
     Follow-up (same session_id): "What tests? What edge cases to replicate?"
No-MCP: python main.py query --repo-path ... --query same questions.
 
### Step 4 — Produce plan (present before coding)
Feature: [Name]
Summary: [One paragraph]
Analogous Feature: file (from CodeGraph), symbol (from CodeGraph), why analogous
Files to Create: path — purpose
Files to Modify: path — what changes (verified by blast radius)
Files NOT to Touch: [prevents scope creep]
Implementation Steps: [with exact file + function from CodeGraph]
Tests Required: [test files from CodeGraph]
Risks: [from blast radius + FastCode]
 
### Step 5 — Wait for approval
Do not write any code until user approves.
If changes requested: revise plan and re-present.
"@
 
Write-Host "   Skills written." -ForegroundColor Green
Write-Host ""
 
# ── Step 6: Summary ───────────────────────────────────────────────────────────
Write-Host "[6/6] Done." -ForegroundColor Green
Write-Host ""
Write-Host "  Project  : $ProjectPath"
Write-Host "  Mode     : $(if ($NoMCP) { 'No-MCP (Bash CLI + Hooks)' } else { 'MCP' })"
Write-Host "  CodeGraph: initialized (.codegraph/ present)"
Write-Host "  CLAUDE.md: written"
Write-Host "  Skills   : repo-navigation, documentation-writer, tutorial-writer,"
Write-Host "             security-review, feature-planner"
Write-Host "  Hooks    : .claude/settings.json written"
Write-Host "  Docs     : docs/ and docs/TUTORIALS/ created"
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
if ($NoMCP) {
    Write-Host "  1. Review and customize CLAUDE.md for this project."
    Write-Host "  2. Run: claude"
    Write-Host "  3. Test: Use Bash to run 'codegraph query \"main\"'"
    Write-Host "  4. Test: /skills  (confirm 5 skills listed)"
} else {
    Write-Host "  1. Review and customize CLAUDE.md for this project."
    Write-Host "  2. Run: claude"
    Write-Host "  3. Inside Claude Code: /mcp  (confirm codegraph + fastcode)"
    Write-Host "  4. Inside Claude Code: /skills  (confirm 5 skills listed)"
    Write-Host "  5. Test: Use codegraph_search to find the main entry point."
}
Write-Host ""
Write-Host "Re-index after major changes : codegraph index" -ForegroundColor Gray
Write-Host "Update Claude Code           : winget upgrade Anthropic.ClaudeCode" -ForegroundColor Gray
```
 
**Usage:**
```powershell
# Standard MCP setup
.\setup-claude-project.ps1 "C:\repo\my-project"
 
# Enterprise / MCP blocked
.\setup-claude-project.ps1 "C:\repo\my-project" -NoMCP
```
 
---
 
## Section 7 — Verify After Bootstrap
 
### MCP path
```
claude
/mcp      ← codegraph and fastcode must appear
/skills   ← repo-navigation, documentation-writer, tutorial-writer,
             security-review, feature-planner must appear
```
 
Test correct exploration pattern:
```
Use codegraph_search to find the main entry point.
Then spawn an Explore agent to explain what it does.
Do NOT call codegraph_explore from this main session.
```
 
### No-MCP path
```powershell
# Verify CodeGraph CLI works
codegraph query "main"
codegraph status
 
# Verify FastCode CLI works
cd C:\tools\FastCode
.\.venv\Scripts\python.exe main.py query `
  --repo-path "C:\repo\my-project" `
  --query "What is the main entry point?"
```
 
Inside Claude Code:
```
/skills    ← confirm 5 skills listed
Use Bash to run: codegraph files "authentication"
```
 
---
 
## Section 8 — Daily Usage
 
### Start of session (MCP)
```
/mcp                    ← verify both servers
list_repos              ← confirm FastCode has this repo indexed
list_sessions           ← check for prior sessions to reuse
```
 
### Start of session (No-MCP)
```powershell
codegraph status        # confirm Backend: native
```
 
### Explore correctly (MCP)
```
Main session: Use codegraph_search to find [symbol].
Then: Spawn an Explore agent — use codegraph_explore to explain how [feature] works.
```
 
### Explore correctly (No-MCP)
```
Use Bash to run: codegraph context "how does [feature] work"
```
 
### Feature work
```
/feature-planner I need to implement [FEATURE].
Find the analogous pattern with CodeGraph first.
Produce a plan and wait for my approval before writing code.
```
 
### After large refactor
```powershell
codegraph index
# MCP: inside Claude Code — Use remove_repo then code_qa to refresh FastCode index
# No-MCP: FastCode re-indexes automatically on next CLI query
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
| FastCode stale index (MCP) | `remove_repo` then `code_qa` in Claude Code |
| FastCode CLI error (No-MCP) | Activate venv first: `.\.venv\Scripts\Activate.ps1` then retry |
| `codegraph init` says already initialized | Run `codegraph index` instead |
| Hooks not firing | Check `.claude/settings.json` is valid JSON; run `/hooks` inside Claude Code |
 
---
 
## Quick Reference Card
 
```
ONCE EVER:
  npm install -g @colbymchenry/codegraph
  npx @colbymchenry/codegraph            ← global MCP config
  git clone FastCode + uv venv + install ← FastCode setup
 
PER NEW PROJECT:
  .\setup-claude-project.ps1 "C:\path\to\project"          ← MCP
  .\setup-claude-project.ps1 "C:\path\to\project" -NoMCP   ← No-MCP
 
AFTER MAJOR CHANGES:
  codegraph index                        ← refresh CodeGraph
  remove_repo + code_qa (MCP)            ← refresh FastCode
  FastCode CLI re-indexes automatically (No-MCP)
 
─────────────────────────────────────────────────────────────────
CODEGRAPH RULES (from official README):
  Main session:       codegraph_search, codegraph_callers,
                      codegraph_callees, codegraph_impact
  Explore agent only: codegraph_explore, codegraph_context
 
CODEGRAPH CLI (No-MCP):
  codegraph query "symbol"        → find symbol (= codegraph_search)
  codegraph files "topic"         → relevant files
  codegraph context "task"        → full context (≈ codegraph_context via Bash)
  codegraph affected "file"       → blast radius (≈ codegraph_impact)
  codegraph sync                  → incremental re-index
  codegraph index                 → full re-index
 
FASTCODE MCP TOOLS:
  list_repos / list_sessions → session start, every time
  code_qa (repos, query, session_id) → semantic Q&A
  remove_repo → re-index after refactor
  delete_session → cleanup
 
FASTCODE CLI (No-MCP):
  python main.py query --repo-path "PATH" --query "QUESTION"
 
DECISION ORDER:
  1. CodeGraph lightweight tools or CLI        (free, instant)
  2. Explore subagent / codegraph context CLI  (when source code needed)
  3. FastCode code_qa or CLI                   (when meaning needed)
  4. Read specific identified files
  5. Grep — last resort, specific string only
  NEVER: broad grep as first step
─────────────────────────────────────────────────────────────────
 
Sources:
  CodeGraph  : https://github.com/colbymchenry/codegraph
  FastCode   : https://github.com/HKUDS/FastCode
  Hooks      : https://code.claude.com/docs/en/hooks-guide
  Skills     : https://code.claude.com/docs/en/skills
  Claude Code: https://code.claude.com/docs/en/setup
```
