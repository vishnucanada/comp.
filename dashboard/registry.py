"""Model and training job registries — thread-safe background workers."""

import json
from threading import Lock, Thread

from .config import CHECKPOINTS_DIR
from .helpers import _load_policy


class ModelRegistry:
    """Background-loadable NLPN model cache."""

    def __init__(self):
        self._lock = Lock()
        self._models: dict[str, object] = {}
        self._toks: dict[str, object] = {}
        self._status: dict[str, str] = {}

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
            from src.enforcer import detect_rmax, load_model, load_nlpn, wrap_with_nlpn

            ckpt = CHECKPOINTS_DIR / name
            cfg = json.loads((ckpt / "nlpn_config.json").read_text())
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
                self._toks[name] = tokenizer
                self._status[name] = "ready"
        except Exception as e:
            with self._lock:
                self._status[name] = f"error:{e}"


class TrainingRegistry:
    """Background training jobs."""

    def __init__(self):
        self._lock: Lock = Lock()
        self._status: dict[str, str] = {}
        self._progress: dict[str, list[dict]] = {}

    def status(self, name: str) -> str:
        return self._status.get(name, "not_started")

    def all_status(self) -> dict[str, str]:
        return dict(self._status)

    def progress(self, name: str) -> list[dict]:
        with self._lock:
            return list(self._progress.get(name, []))

    def train_async(self, name: str, config: dict) -> None:
        with self._lock:
            if self._status.get(name) == "training":
                return
            self._status[name] = "training"
            self._progress[name] = []
        Thread(target=self._train, args=(name, config), daemon=True).start()

    def _train(self, name: str, config: dict) -> None:
        try:
            import src
            from src.enforcer import detect_rmax, load_model, wrap_with_nlpn
            from src.train import TrainConfig, build_deny_examples, calibrate_privilege

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

            model, tokenizer = load_model(model_id)
            rmax = detect_rmax(model)
            wrap_with_nlpn(model, rmax=rmax)

            deny_ex = build_deny_examples(policy)

            def on_step(epoch: int, step: int, loss: float) -> None:
                with self._lock:
                    self._progress[name].append(
                        {"epoch": epoch, "step": step, "loss": round(loss, 6)}
                    )

            src.train_nlpn(
                model,
                tokenizer,
                policy,
                config=TrainConfig(
                    epochs=config.get("epochs", 3),
                    lr=config.get("lr", 1e-4),
                    orth_reg=config.get("orth_reg", 0.0),
                ),
                deny_examples=deny_ex,
                on_step=on_step,
            )

            low_g = calibrate_privilege(model, tokenizer, deny_ex, rmax=rmax, policy=policy)

            CHECKPOINTS_DIR.mkdir(exist_ok=True)
            src.save_nlpn(model, ckpt, model_id=model_id, low_g=low_g)

            with self._lock:
                self._status[name] = "done"
        except Exception as e:
            with self._lock:
                self._status[name] = f"error:{e}"


model_registry = ModelRegistry()
training_registry = TrainingRegistry()
