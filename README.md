# comp. — Nested Least-Privilege Networks

Policy-enforced LLM deployment via rank-restricted transformer layers. Instead of filtering outputs, comp. constrains what the model can *compute* at low privilege — capabilities become structurally unreachable rather than merely blocked.

```
Policy text  →  Translate  →  Monitor  →  Allocate  →  Enforce
                (NLP/LLM)   (check prompt)  (set g)   (rank-restrict)
```

---

## How it works

Each `nn.Linear` layer in the model is replaced with an **NLPNLinear** layer:

```
W(g) = B[:, :g] @ A[:g, :]          g ∈ [1, rmax]
```

At privilege `g`, only the first `g` rank-1 components of the weight matrix are active. The nested structure guarantees `Im(W(g)) ⊆ Im(W(g+1))` — lower privilege is a strict subset of higher privilege, so capability reduction is structural, not a filter.

**Training** fine-tunes only the `B` matrices so that at `low_g` the model genuinely refuses denied topics, while at `rmax` normal behaviour is preserved. After training, `calibrate_privilege` binary-searches for the highest `g` that still achieves a target suppression rate — maximising capability while maintaining the enforcement guarantee.

---

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python ≥ 3.11. Optional: `bitsandbytes` for 4-bit/8-bit quantisation on CUDA.

---

## Quick start (CLI)

### 1. Write a policy

```
name: HR Compliance

DENY: salary information
  match: salary, wage, compensation, how much does

DENY: personal contact
  match: address, phone, email

ALLOW: general information
```

Save as `policies/hr.txt`. Or write plain English and let the translator extract the rules:

```bash
python -c "
from src.translator import PolicyTranslator
p = PolicyTranslator().translate('Employee salaries must not be shared. Phone numbers are confidential.')
print(p.to_text())
"
```

The translator tries Anthropic Claude, then a local Ollama model, then falls back to rule-based NLP — no API key required for the fallback.

### 2. Train

```bash
comp train policies/hr.txt --model Qwen/Qwen2.5-0.5B --epochs 3 --calibrate
```

`--calibrate` runs a binary search after training to find the highest privilege level that still suppresses ≥ 90 % of denied prompts. The calibrated `low_g` is saved to the checkpoint and used automatically at inference.

Checkpoint is written to `nlpn_checkpoints/hr/`.

### 3. Evaluate

```bash
comp eval policies/hr.txt nlpn_checkpoints/hr/ --model Qwen/Qwen2.5-0.5B
```

Prints `deny_suppression_rate` and `allow_preservation_rate`.

### 4. Dashboard

```bash
comp serve
# or: uvicorn dashboard:app --reload
```

Opens at `http://localhost:8000`. Manage policies, trigger training jobs (with live step-by-step loss via `GET /api/train/{name}/status`), and chat with privilege-enforced models.

---

## Python API

```python
import src
from src.enforcer import get_device, load_model, load_nlpn, wrap_with_nlpn
from src.policy import PolicyAllocator, PolicyCompiler

device = get_device()
model, tokenizer = load_model("Qwen/Qwen2.5-0.5B", device)
rmax = src.detect_rmax(model)
src.wrap_with_nlpn(model, rmax=rmax)
src.load_nlpn(model, "nlpn_checkpoints/hr")

policy = src.Policy.from_file("policies/hr.txt")
allocator = PolicyAllocator(PolicyCompiler(policy), tokenizer, low_privilege=3)

enc = tokenizer("What is John's salary?", return_tensors="pt")
output_ids, g_used = allocator.generate(model, enc["input_ids"], rmax=rmax, max_new_tokens=50)
# g_used == 3 → low privilege → model refuses
```

### Training

```python
from src.train import TrainConfig, build_deny_examples, evaluate_nlpn, calibrate_privilege

deny_ex = build_deny_examples(policy)
src.train_nlpn(model, tokenizer, policy, config=TrainConfig(epochs=3, lr=1e-4))

metrics = evaluate_nlpn(model, tokenizer, deny_ex, allow_examples, rmax=rmax, policy=policy)
print(metrics)
# {'deny_suppression_rate': 0.92, 'allow_preservation_rate': 0.90, 'low_g': 44, 'rmax': 896}

low_g = calibrate_privilege(model, tokenizer, deny_ex, rmax=rmax, policy=policy)
src.save_nlpn(model, "nlpn_checkpoints/hr", model_id="Qwen/Qwen2.5-0.5B", low_g=low_g)
```

### GDPR mode

```python
from src.gdpr import GDPRPolicyParser, GDPRAllocator, AuditLog, verify_audit_log
import os

rules, name = GDPRPolicyParser.parse(open("policies/gdpr.txt").read())
audit = AuditLog("audit/gdpr_audit.jsonl", hmac_key=os.environ["GDPR_AUDIT_HMAC_KEY"].encode())
allocator = GDPRAllocator(rules, tokenizer, audit_log=audit)

result = verify_audit_log("audit/gdpr_audit.jsonl", hmac_key=...)
print(result)  # {'total': 100, 'valid': 100, 'tampered': 0, 'unsigned': 0}
```

---

## How suppression is detected

`_is_suppressed(text, denied_keywords)` uses content-first logic:

1. If output contains a denied keyword → information leaked → **not suppressed**
2. If output contains a refusal phrase → **suppressed**
3. Empty output → **suppressed**
4. Otherwise → **not suppressed**

This avoids false positives from short outputs and false negatives from polite-but-leaky responses ("The salary is $80,000 but I shouldn't share this").

---

## Configuration

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Enables Claude as LLM backend (chat + policy translation) |
| `API_KEY` | Protects write endpoints — unset to disable auth |
| `ALLOWED_ORIGINS` | Comma-separated CORS origins (default: `http://localhost:8000`) |
| `GDPR_AUDIT_HMAC_KEY` | Enables tamper-evident GDPR audit log signing |

---

## Project structure

```
src/
  nlpn.py          NLPNLinear — rank-restricted linear layer (core primitive)
  enforcer.py      wrap_with_nlpn, set_privilege, save_nlpn, load_nlpn
  policy.py        Policy, PolicyCompiler, PolicyAllocator
  translator.py    Plain-English → Policy (Anthropic → Ollama → rule-based)
  gdpr.py          GDPRAllocator, AuditLog, verify_audit_log
  train.py         train_nlpn, evaluate_nlpn, calibrate_privilege
  cli.py           `comp` CLI entry point

dashboard/
  app.py           FastAPI application
  backends.py      Ollama, Anthropic, NLPN generation backends
  registry.py      ModelRegistry, TrainingRegistry (background threads, progress tracking)
  routers/         REST endpoints (chat, policies, training, models, admin)
  static/          Frontend HTML

tests/
  test_e2e.py      End-to-end pipeline tests (no network/GPU required)
  test_nlpn.py     NLPNLinear unit tests

policies/          Example policy files
nlpn_checkpoints/  Saved NLPN checkpoints (A/B matrices + calibrated low_g)
```

## Running tests

```bash
pytest tests/ -v
```

No network or GPU required — tests use a tiny in-process model.
