---
name: push-changes
description: Commit and push pending working-tree changes to the remote, split into a reasonable number of logically-grouped commits based on what actually changed. Handles pre-commit hook failures by fixing root causes, never by bypassing them. Never sets up or touches CI/CD pipelines. Use when the user asks to push changes, commit and push, or sync the working tree to remote.
disable-model-invocation: true
---

# Push Changes

User-invoked workflow (`/push-changes`) for turning pending working-tree
changes into a small number of coherent commits and pushing them to the
configured remote. This exists specifically so pushes don't happen as a
side effect of some other task — only when the user explicitly asks.

## Scope — read this first

This skill's job is committing and pushing. It must **never** create,
modify, or propose CI/CD pipeline configuration (GitHub Actions, GitLab
CI, pre-commit hook *installation*, etc.) as part of this flow, even if a
change seems to call for one. If the diff suggests CI/CD is missing or
broken, mention it to the user as an aside and stop there — do not act on
it under this skill.

## Steps

1. **Survey the change set**
   - `git status` (never `-uall`) for the full untracked/modified/deleted
     picture.
   - `git diff` and `git diff --staged` to see actual content changes.
   - `git log --oneline -10` to match this repo's existing commit message
     voice and level of detail.

2. **Decide the commit grouping**
   - Don't default to one mega-commit, and don't default to one commit per
     file either. Group by logical concern — e.g. a spec/doc change, a
     source module together with its tests, a dependency/config bump, a
     memory/instruction file update — each as its own commit.
   - "Reasonable number" means: one commit per independent concern actually
     present in the diff. If the whole diff is one tightly coupled change,
     one commit is correct. If it spans clearly separate concerns (e.g.
     "implemented feature X" and "fixed unrelated typo in README"), split
     them.
   - Never split a single atomic change (e.g. a function and its call
     site, or a schema change and the code that depends on it) across
     commits — every commit should leave the repo in a working state.
   - Stage explicit paths per group (`git add <specific paths>`) rather
     than `git add -A`/`git add .`, so unrelated or accidental files never
     ride along with a group they don't belong to.

3. **Check staged content before each commit**
   - After staging a group, run `git status` / `git diff --staged` and
     scan for anything that looks like a secret, credential, or a file
     that shouldn't be committed — even if the filename looks innocuous.
   - If something suspicious is staged, unstage it (`git restore --staged
     <file>`), flag it to the user explicitly, and do not commit it
     silently.

4. **Commit each group**
   - Write a message that explains *why*, not just *what*, matching the
     repo's existing style (check recent `git log` messages for tone).
   - End every commit message with whatever attribution line is currently
     in effect for this session per the active system instructions — check
     for it fresh each time rather than assuming a fixed line, since that
     guidance can change between sessions/models.

5. **Handle pre-commit hooks properly**
   - A hook that auto-formats files or fails on lint/type errors is real
     signal. Investigate and fix the root cause in the actual files, then
     re-stage and re-commit.
   - Never bypass a hook with `--no-verify`, and never bypass signing with
     `--no-gpg-sign` or `-c commit.gpgsign=false`.
   - If a hook failure reflects a real design/logic issue rather than a
     formatting nit, stop and explain it to the user instead of forcing
     the commit through.
   - If a fix attempt doesn't resolve the failure, try once more with a
     clearer diagnosis; if still stuck, surface it to the user rather than
     looping.

6. **Push**
   - Plain `git push` to the current branch's configured upstream. Do not
     force-push, and do not push to a different branch than the one
     currently checked out, unless the user explicitly said otherwise in
     this conversation.
   - If the push is rejected because the remote has commits this branch
     doesn't have, don't force it through — fetch, then merge or rebase
     (whichever matches how this repo has handled that before, or ask if
     unclear), resolve any conflicts, and retry. Never discard local or
     remote commits to force a push through.

7. **Report back**
   - List the commits actually created (short hash + one-line summary
     each) and confirm the push succeeded, naming the remote and branch.
   - If anything was deliberately left uncommitted (a flagged secret, an
     unrelated in-progress edit), say so explicitly rather than letting it
     pass silently.

## Out of scope

- Setting up, modifying, or suggesting CI/CD pipelines — never part of
  this skill, even as a "while we're here" addition.
- Force-push, history rewriting, or branch deletion — if genuinely needed,
  that's a separate explicit conversation with the user, not something
  this skill does on its own judgment.
