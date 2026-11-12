# What EV lets a coding agent do

These are the rules behind `config/agent_rules.toml`. The TOML file is what runs; this page is the reason for each line, in plain
words, so the rules can be argued with and changed on purpose. EV never decides these from what an agent, a web page or a file
says. It decides from the request itself.

1. **Nothing gets deleted.** No removing files or folders, no `git clean`, no hard resets, no force pushes, no dropping stashes.
   If something genuinely has to go, the agent says so and you do it.
2. **Nothing that runs unseen code from the network.** No `curl | sh`, no `Invoke-Expression`, no encoded PowerShell.
3. **Credentials are off limits.** SSH keys, cloud credentials, tokens, browser data, `.npmrc`/`.pypirc`. The agent doesn't read or
   write them, and doesn't get asked about them.
4. **The agent can't switch its own supervisor off.** Its settings files, hook configs and these rules are refused for edits.
5. **Reading and checking inside the project is fine.** Looking at files, `git status/diff/log`, tests, linters and type checks
   are allowed without a word.
6. **Editing is fine inside the project folder only.** Writing anywhere else is a question, not a yes.
7. **Anything else is a question, asked out loud.** Installs, commits, builds, network calls, unknown commands, odd paths, tools EV
   has never seen. One short sentence; you answer yes or no. No answer means the agent shows its own prompt.
8. **You can open a door for a while, never take down a wall.** "Let Claude install packages for an hour" lets one kind of thing
   through, in the project it is working in, until the time runs out. It cannot reach rules 1 to 4.
9. **Use current, maintained technology.** With the package check on, an install of something with a known vulnerability or a
   yanked release is refused, and a package that is missing, deprecated, abandoned or installed from a URL is asked about. The check
   reads registry data (PyPI, npm, OSV), not web pages, and it can only make an answer stricter.
10. **If EV can't be sure, it steps back.** Paused, offline, unsure what a command does: no decision, and the agent's own prompt
    appears. EV is never the reason something got approved by accident.

Every decision, with the reason and who made it (rules or you), is written to `data/agent_audit.jsonl` and shown in the dashboard.
