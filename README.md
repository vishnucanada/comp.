# comp. — Nested Least-Privilege Networks

Policy-enforced LLM deployment via rank-restricted transformer layers. Instead of filtering outputs, comp. constrains what the model can *compute* at low privilege — capabilities become structurally unreachable rather than merely blocked.

```
Policy text  →  Translate  →  Monitor  →  Allocate  →  Enforce
                (NLP rules)  (check prompt)  (set g)   (rank-restrict)
```

---

## Quick start

```bash
# 1. Install
pip install -e ".[dashboard]"

# 2. Copy env template and add your Anthropic key (optional)
cp .env.example .env

# 3. Start the dashboard
uvicorn dashboard:app --reload

# 4. Open http://localhost:8000
```

To also run local model inference, start Ollama in a separate terminal:
```bash
ollama pull qwen2.5:0.5b && ollama serve
```

---

## How it works

Each `nn.Linear` layer in the model is replaced with an **NLPNLinear** layer:

```
W(g) = B[:, :g] @ A[:g, :]          g ∈ [1, rmax]
```

At privilege `g`, only the first `g` rank-1 components of the weight matrix are active. The nested structure guarantees `Im(W(g)) ⊆ Im(W(g+1))` — lower privilege is a strict subset of higher privilege.

**Training** (Algorithm 1) fine-tunes only the `B` matrices so that at `low_g` the model genuinely refuses denied topics, while at `rmax` normal behaviour is preserved.

---

## Policy format

```
name: HR Compliance

DENY: salary information
  match: salary, wage, compensation, pay
  regex: \$[\d,]+

DENY: medical information
  match: medical, health, diagnosis

ALLOW: general information
```

Natural language policies are also accepted — the `PolicyTranslator` extracts rules automatically:

```python
from src import PolicyTranslator
policy = PolicyTranslator().translate("""
    Employee salaries and medical records must not be disclosed.
    Job titles and public company information may be shared.
""")
```

---

## Python API

```python
import src

# Load and wrap a model
model, tokenizer = src.utils.load_model("Qwen/Qwen2.5-0.5B")
rmax  = src.detect_rmax(model)
src.wrap_with_nlpn(model, rmax=rmax)

# Define a policy
policy   = src.Policy.from_file("policies/legal_compliance.txt")
compiler = src.PolicyCompiler(policy)
low_g    = max(1, rmax // 20)
allocator = src.PolicyAllocator(compiler, tokenizer, low_privilege=low_g)

# Generate with privilege enforcement
enc = tokenizer("What is John's salary?", return_tensors="pt")
output_ids, g_used = allocator.generate(model, enc.input_ids, rmax=rmax, max_new_tokens=50)
```

### Training

```python
from src.train import TrainConfig

config = TrainConfig(epochs=3, lr=1e-4)
src.train_nlpn(model, tokenizer, policy, config=config)

# Measure how well suppression worked
from src.train import evaluate_nlpn, build_deny_examples
deny_ex  = build_deny_examples(policy)
metrics  = evaluate_nlpn(model, tokenizer, deny_ex, src.utils._DEFAULT_ALLOW)
print(metrics)
# {'deny_suppression_rate': 0.9, 'allow_preservation_rate': 1.0, 'low_g': 44, 'rmax': 896}

# Find the optimal low_g automatically
from src.train import calibrate_privilege
low_g = calibrate_privilege(model, tokenizer, deny_ex, rmax=rmax)
```

### Save / load

```python
src.save_nlpn(model, "nlpn_checkpoints/hr_policy", model_id="Qwen/Qwen2.5-0.5B")

# Later, in a new session:
model, tokenizer = src.utils.load_model("Qwen/Qwen2.5-0.5B")
src.wrap_with_nlpn(model, rmax=rmax)
src.load_nlpn(model, "nlpn_checkpoints/hr_policy")
```

### GDPR mode

```python
from src.gdpr import GDPRPolicyParser, GDPRAllocator, AuditLog, verify_audit_log
import os

rules, name = GDPRPolicyParser.parse(open("policies/gdpr.txt").read())
audit = AuditLog("audit/gdpr_audit.jsonl", hmac_key=os.environ["GDPR_AUDIT_HMAC_KEY"].encode())
allocator = GDPRAllocator(rules, tokenizer, audit_log=audit)

# Later, verify log integrity
result = verify_audit_log("audit/gdpr_audit.jsonl", hmac_key=...)
print(result)  # {'total': 100, 'valid': 100, 'tampered': 0, 'unsigned': 0}
```

---

## Dashboard

The web dashboard (`/dashboard`) provides:
- **Policy editor** — create and test policies with live accuracy scoring
- **Chat** — test enforcement interactively; uses the NLPN model if one is loaded, otherwise falls back to Ollama/Claude
- **Model loader** — `POST /api/models/load/{policy_name}` to load a trained checkpoint in the background
- **Policy history** — every save is versioned; `GET /api/policies/{name}/history` lists versions

### Loading a trained model into the dashboard

```bash
# 1. Train and save
python demo_train.py          # saves to nlpn_checkpoints/legal_compliance/

# 2. Start dashboard
uvicorn dashboard:app

# 3. Load the model (runs in background, takes 1-5 min)
curl -X POST http://localhost:8000/api/models/load/legal_compliance \
     -H "X-API-Key: $API_KEY"

# 4. Check status
curl http://localhost:8000/api/models/status

# 5. Chat now uses rank-restricted generation
```

### Streaming chat

`POST /api/chat/stream` returns Server-Sent Events:

```
data: {"type": "policy", "decision": "ALLOW", "violations": []}
data: {"type": "chunk", "text": "The capital of France "}
data: {"type": "chunk", "text": "is Paris."}
data: {"type": "done", "model": "ollama/qwen2.5:0.5b"}
```

---

## Configuration

Copy `.env.example` to `.env`:

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Enables Claude as a fallback LLM backend |
| `API_KEY` | Protects write endpoints — unset to disable auth |
| `ALLOWED_ORIGINS` | Comma-separated CORS origins (default: `http://localhost:8000`) |
| `GDPR_AUDIT_HMAC_KEY` | Enables tamper-evident GDPR audit log signing |

---

## Docker

```bash
docker compose up
```

Starts the dashboard on port 8000 and Ollama on 11434. Pull a model after first start:

```bash
docker compose exec ollama ollama pull qwen2.5:0.5b
```

---

## Development

```bash
pip install -e ".[dev]"
pytest                  # run all tests
pytest tests/test_integration.py -v   # integration tests only
```

### Project structure

```
src/
  nlpn.py          NLPNLinear — rank-restricted linear layer (core primitive)
  enforcer.py      wrap_with_nlpn, set_privilege, save_nlpn, load_nlpn
  policy.py        Policy, PolicyCompiler, PolicyAllocator, PolicyStack
  semantic.py      SemanticPolicyCompiler (embedding-based matching)
  translator.py    Plain-English → Policy via Claude
  gdpr.py          GDPRAllocator, AuditLog, verify_audit_log
  train.py         train_nlpn, evaluate_nlpn, calibrate_privilege
  utils.py         get_device, load_model
  cli.py           `comp` CLI entrypoint

dashboard/
  __init__.py      FastAPI app — uvicorn dashboard:app
  config.py        Constants and directory paths
  deps.py          Rate limiter and API-key auth dependency
  schemas.py       Pydantic request models
  helpers.py       sanitize, _persist_policy, _load_policy
  registry.py      ModelRegistry, TrainingRegistry (background threads)
  backends.py      Ollama, Anthropic, NLPN generation, SSE streaming
  routers/
    chat.py        POST /api/chat  and  POST /api/chat/stream
    models.py      GET/POST /api/models/...
    policies.py    CRUD /api/policies/... and POST /api/enact
    training.py    POST/GET /api/train/...
  static/
    dashboard.html
    landing.html
    favicon.png

examples/
  demo_policy.py   End-to-end enforcement demo
  demo_gdpr.py     GDPR tiered enforcement demo
  demo_train.py    Training + save demo

tests/             Unit + integration tests
policies/          Example policy files
```
