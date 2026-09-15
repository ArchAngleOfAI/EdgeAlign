# AGENT.md — Operating Instructions for the EdgeAlign Coding Agent

This file defines how the Claude Code agent should behave in this repository.
It was established by the project owner on 2026-09-14 and takes precedence
over the agent's generic defaults whenever the two conflict.

## Core directive

- Stay faithful to the user's prompts and stated context. Implement the
  user's ideas as given.
- Do not improvise or substitute a "more conventional" approach without the
  user's explicit consent, even when a design in this project looks unusual
  compared to standard practice. The user is aware their approach may
  contradict conventional methods — that is expected, not a mistake to
  silently correct.
- If something in a spec looks like it might not work, or conflicts with
  another spec/decision already on record, say so and ask — don't quietly
  route around it.

## Memory system

Three persistent files, maintained by the agent, live at the repo root:

- **AGENT.md** (this file) — agent personality, standing instructions, and
  the operating rules the user has given. Update this file when the user
  gives a new standing instruction about *how* to work (not project facts).
- **MEMORY.md** — long-term project memory: goal, architecture, spec
  history, repo/git structure, dataset choices, decisions and their
  rationale. This is the durable knowledge base for the project.
- **SHORT_MEMORY.md** — short-term / working memory: what's in progress
  right now, open threads, immediate next steps. Written so that if the
  agent session is interrupted or restarted, a fresh session can read this
  file and pick up where things left off without re-deriving context.

Keep MEMORY.md and SHORT_MEMORY.md current as work happens — don't let them
drift stale. SHORT_MEMORY.md should be pruned/updated as tasks complete;
MEMORY.md accumulates durable facts and should be edited (not just appended)
when old entries become outdated.

## Git discipline

- Maintain the git repository properly: meaningful commits, nothing
  sensitive staged, working tree kept clean of accidental cruft.
- Do not commit automatically. After making changes, notify the user that
  there are changes ready to be committed and let them decide when/how to
  commit, unless the user has explicitly told the agent to commit on its
  own for a given task.
- Follow standard git safety practice: check `git status`/`git diff` before
  any destructive operation, never force-push or rewrite history without
  explicit approval.
- When the user wants pending changes committed and pushed, use the
  `/push-changes` skill (`.claude/skills/push-changes/SKILL.md`) — it
  groups the diff into a reasonable number of logical commits, handles
  pre-commit hook failures by fixing root causes (never bypassing them),
  and pushes. It deliberately never sets up or touches CI/CD pipelines.

## Reporting implementation work

- Each time a major implementation milestone is completed (a new
  architecture variant stood up, a training loop working end-to-end, a
  significant refactor, etc.), explain it clearly to the user: what was
  built, how it works, and how it maps back to the relevant spec(s) in
  `PROMPTS/`. Don't just say "done" — walk through it.
- Maintain a proper `README.md` **for each branch of the project** (e.g. a
  branch implementing a specific architecture variant, or a specific
  experiment line). Each branch's README should describe what that branch
  is doing, its current state, and how it differs from other branches/the
  main line, kept up to date as work on that branch progresses.

## Project identity

- Project name: EdgeAlign.
- See MEMORY.md for the actual project goal and technical content.
