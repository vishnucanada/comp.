"""Model and training job registries — thread-safe background workers."""
import json
import queue
from pathlib import Path
from threading import Lock, Thread

from .config import CHECKPOINTS_DIR
from .helpers import _load_policy


class ModelRegistry:
    """Background-loadable NLPN model cache."""

    def __init__(self):
        self._lock   = Lock()
        self._models: dict[str, object] = {}
        self._toks:   dict[str, object] = {}
        self._status: dict[str, str]    = {}

    def status(self, name: str) -> str:
        return self._status.get(name, "not_loaded")

    def all_status(self) -> dict[str, str]:
        return dict(self._status)

    def get(self, name: str):
        with self._lock:
            if name in self._models:
                return self._models[name], self._toks[name]
        return None, None

    def load_async(self, name: str) -> None:
        with self._lock:
            if self._status.get(name) in ("loading", "ready"):
                return
            self._status[name] = "loading"
        Thread(target=self._load, args=(name,), daemon=True).start()

    def _load(self, name: str) -> None:
        try:
            from src.utils import load_model
            from src.enforcer import detect_rmax, load_nlpn, wrap_with_nlpn

            ckpt = CHECKPOINTS_DIR / name
            cfg  = json.loads((ckpt / "nlpn_config.json").read_text())
            model_id = cfg.get("model_id")
            if not model_id:
                raise ValueError("nlpn_config.json missing model_id")

            model, tokenizer = load_model(model_id)
            leaf_names = list({n.split(".")[-1] for n in cfg.get("layers", {})})
            wrap_with_nlpn(model, rmax=detect_rmax(model), target_modules=leaf_names)
            load_nlpn(model, ckpt)
            model.eval()

            with self._lock:
                self._models[name] = model
                self._toks[name]   = tokenizer
                self._status[name] = "ready"
        except Exception as e:
            with self._lock:
                self._status[name] = f"error:{e}"


class TrainingRegistry:
    """Background training jobs with live SSE progress streaming."""

    def __init__(self):
        self._lock:   Lock                   = Lock()
        self._status: dict[str, str]         = {}
        self._queues: dict[str, queue.Queue] = {}

    def status(self, name: str) -> str:
        return self._status.get(name, "not_started")

    def all_status(self) -> dict[str, str]:
        return dict(self._status)

    def stream_queue(self, name: str) -> queue.Queue | None:
        return self._queues.get(name)

    def train_async(self, name: str, config: dict) -> None:
        with self._lock:
            if self._status.get(name) == "training":
                return
            self._status[name] = "training"
            self._queues[name] = queue.Queue()
        Thread(target=self._train, args=(name, config), daemon=True).start()

    def _train(self, name: str, config: dict) -> None:
        q = self._queues[name]
        try:
            import src
            from src.enforcer import detect_rmax, wrap_with_nlpn
            from src.train import (
                TrainConfig, build_adversarial_examples, build_deny_examples,
            )
            from src.utils import load_model

            policy = _load_policy(name)
            if policy is None:
                raise ValueError(f"Policy '{name}' not found")

            model_id = config.get("model_id", "Qwen/Qwen2.5-0.5B")
            ckpt = CHECKPOINTS_DIR / name
            if ckpt.exists():
                try:
                    saved = json.loads((ckpt / "nlpn_config.json").read_text()).get("model_id")
                    if saved:
                        model_id = saved
                except Exception:
                    pass

            q.put({"type": "status", "message": f"Loading {model_id} ..."})
            model, tokenizer = load_model(model_id)
            wrap_with_nlpn(model, rmax=detect_rmax(model))

            deny_ex = build_deny_examples(policy)
            if config.get("adversarial"):
                deny_ex += build_adversarial_examples(policy)

            def on_step(epoch, step, loss):
                q.put({"type": "progress", "epoch": epoch, "step": step,
                       "loss": round(loss, 4)})

            q.put({"type": "status", "message": "Training ..."})
            src.train_nlpn(
                model, tokenizer, policy,
                config=TrainConfig(
                    epochs=config.get("epochs", 3),
                    lr=config.get("lr", 1e-4),
                    orth_reg=config.get("orth_reg", 0.0),
                ),
                deny_examples=deny_ex,
                on_step=on_step,
            )

            q.put({"type": "status", "message": "Saving checkpoint ..."})
            CHECKPOINTS_DIR.mkdir(exist_ok=True)
            src.save_nlpn(model, ckpt, model_id=model_id)

            with self._lock:
                self._status[name] = "done"
            q.put({"type": "done", "checkpoint": str(ckpt)})
        except Exception as e:
            with self._lock:
                self._status[name] = f"error:{e}"
            q.put({"type": "error", "message": str(e)})
        finally:
            q.put(None)


model_registry    = ModelRegistry()
training_registry = TrainingRegistry()
