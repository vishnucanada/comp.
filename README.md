# comp. — Policy-as-Code for LLM API Calls

A small enforcement layer for any LLM backend. Define DENY/ALLOW rules in a
text policy, plug in a content guard, and route every request through
`PolicyGate`. Every decision is appended to a JSONL audit log.

```
prompt + role  →  PolicyGate  →  LLM call (or deny)  →  audit log
                  (Guard + IAM)
```

The core library has zero runtime dependencies. Guard backends and the
dashboard are opt-in extras.

---

## Install

```bash
pip install -e .
```

Optional extras:

| Extra              | What it adds                                                |
| ------------------ | ----------------------------------------------------------- |
| `comp[openai]`     | `OpenAIModerationGuard` (free, OPENAI_API_KEY required)     |
| `comp[llamaguard]` | `LlamaGuardGuard` — self-hosted Meta Llama Guard 3          |
| `comp[yaml]`       | YAML IAM configs                                            |
| `comp[dashboard]`  | FastAPI web UI (`comp serve`)                               |

---

## Quick start

### 1. Write a policy

```
name: HR Compliance

DENY: salary information
  match: salary, wage, compensation
  regex: how\s+much\s+does

DENY: personal contact
  match: address, phone, email

ALLOW: general information
```

Save as `policies/hr.txt`.

### 2. Gate any LLM call

```python
from anthropic import Anthropic
from src import IAMConfig, Policy, PolicyGate, make_guard

policy = Policy.from_file("policies/hr.txt")
guard  = make_guard(policy, backend="openai")        # or "keyword", "llamaguard"
iam    = IAMConfig.from_yaml("iam.yaml")
gate   = PolicyGate(policy, guard=guard, iam=iam, audit_log_path="audit/gate.jsonl")

claude = Anthropic()

response, decision = gate.complete(
    message=user_input,
    fn=lambda m: claude.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": m}],
    ).content[0].text,
    user_role="analyst",
)

print(decision)
# GateDecision(DENY['salary information'], role='analyst', via='openai-moderation')
```

If the policy denies the message, the LLM is never called and `gate.deny_message`
is returned instead. The decision is recorded in the audit log either way.

### 3. CLI

```bash
# Single-prompt check from the shell — exit code 0 if allowed, 1 if denied
comp check policies/hr.txt "what is the salary?" --guard openai --role analyst

# Compliance report (markdown or json)
comp report policies/hr.txt --audit-log audit/gate.jsonl --output report.md

# Web dashboard
comp serve
```

---

## Guard backends

| Backend                 | When to use                                         | Cost / deps |
| ----------------------- | --------------------------------------------------- | ----------- |
| `KeywordGuard`          | Cheap fallback. Trivially bypassed by paraphrase.   | stdlib      |
| `OpenAIModerationGuard` | Default for hosted deployments. Managed taxonomy.   | free, key   |
| `LlamaGuardGuard`       | Self-hosted / air-gapped environments.              | torch + 16 GB |
| `CompositeGuard`        | OR over several guards — first hit wins.            | composite   |

Custom guards subclass `Guard` and return a `GuardResult`:

```python
from src.guard import Guard, GuardResult

class MyClassifier(Guard):
    name = "my-classifier"
    def check(self, text, history=None):
        flagged, cats = my_model.predict(text)
        return GuardResult(flagged=flagged, categories=cats, backend=self.name)
```

---

## IAM

Define roles in YAML; the gate uses them for full-privilege bypass and tool
allowlists.

```yaml
default_role: anonymous
roles:
  anonymous:
    privilege: low
    tools: []
  analyst:
    privilege: medium
    tools: [search_docs, get_faq]
  hr_manager:
    privilege: full
    tools: "*"
```

`privilege: full` lets a role bypass content checks entirely. **Unknown role
names never silently inherit `default_role`** — they hit content checking
just like an anonymous request would.

For tool calls:

```python
result, decision = gate.run_tool(
    tool_name="search_docs",
    fn=lambda query: db.search(query),
    args={"query": user_query},
    user_role="analyst",
)
```

---

## Audit log

Every gate decision is appended to a JSONL file:

```json
{"timestamp": "2026-05-07T12:34:56",
 "prompt_hash": "8e9c…",
 "user_role": "analyst",
 "allowed": false,
 "categories": ["salary information"],
 "backend": "openai-moderation"}
```

Prompts are stored as SHA-256 hashes — the original text never lands on disk.
Pipe to your SIEM, or aggregate with `comp report`.

---

## What this is, and isn't

**This is** an operational layer for gating LLM calls: versioned policies,
pluggable detection, role-based bypass, audit trail. It's the kind of
infrastructure you wire into every call site and stop thinking about.

**This isn't** a compliance certification, a jailbreak-proof shield, or a
substitute for a security review. The strength of enforcement depends on the
guard backend you plug in. The keyword guard catches obvious matches; an LLM
classifier guard catches paraphrases; nothing catches everything. Choose the
guard your threat model justifies.

---

## Project layout

```
src/
  policy.py    Policy DSL parser + PolicyCompiler
  guard.py     KeywordGuard, OpenAIModerationGuard, LlamaGuardGuard, CompositeGuard
  iam.py       IAMConfig, Role
  gate.py      PolicyGate (the enforcement object)
  report.py    ComplianceReport + audit-log summary
  cli.py       comp check / comp report / comp serve

dashboard/
  __init__.py  FastAPI app
  routers/     /api/policies, /api/chat, /api/admin, /api/models
  static/      Landing + dashboard HTML

tests/
  test_e2e.py  End-to-end tests — no network, no GPU

policies/      Example policy files
audit/         JSONL decision logs (created on first write)
```

---

## Running tests

```bash
pytest tests/ -v
```

No network, no GPU — the test suite covers policy parsing, the keyword guard,
the IAM layer, the gate, and the report generator. Backend-specific guards
(OpenAI, LlamaGuard) are tested via interface stubs; install the relevant
extras to test them against the real services.
