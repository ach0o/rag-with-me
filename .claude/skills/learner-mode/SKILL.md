---
name: learner-mode
description: Show what to code and where, user types it. Never write code for user. Use when user wants to learn by doing.
allowed-tools: Read, Glob, Grep, Bash, AskUserQuestion, Agent
---

# Learner Mode

## Workflow

### Step 1: Show Full Change List

For every task, produce a numbered list of ALL changes needed. Each item:

- **File path and line number** (or "new file")
- **Action**: add / modify / delete
- **Code**: exact code user should type, in a fenced code block
- **Why**: one-line reason

Example format:

    1. `src/foo.ts:10` — MODIFY
       ```ts
       import { Bar } from './bar';
       ```
       > Need Bar import for usage below.

    2. `src/foo.ts:42` — ADD after line 41
       ```ts
       const bar = new Bar();
       ```
       > Initializes Bar for downstream calls.

Order logically (imports first, then usage, etc). Show COMPLETE list — no partial reveals. Group by file if 10+ items.

### Step 2: Let User Do It

After showing the list, use `AskUserQuestion`:

- Question: "Make these changes, then pick one:"
- Options: **confirm** ("Done and clear") / **do it** ("Just code it for me") / **other** (free text)

### Step 3: Handle Response

- **confirm**: Read changed files, verify edits are correct. If wrong, point out and back to Step 2. If correct, next batch or summarize.
- **do it**: Spawn an Agent to apply all remaining changes from the list. Exit learner mode.
- **other**: Answer/adjust. Re-present updated list if changes needed. Back to Step 2.

## Rules

- Keep explanations short.
- If user is stuck, give hints before answers.
- Always show line numbers.
