from typing import Any, Dict, Optional, Tuple

import torch


def load_checkpoint_payload(path: str, map_location: Any = "cpu") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    payload = torch.load(path, map_location=map_location)
    if isinstance(payload, dict) and "state_dict" in payload:
        state = payload["state_dict"]
        meta = payload.get("meta", {}) if isinstance(payload.get("meta", {}), dict) else {}
        return state, meta
    if isinstance(payload, dict):
        return payload, {}
    raise TypeError(f"Unsupported checkpoint payload type for {path}: {type(payload)!r}")


def can_load_model(model: torch.nn.Module, path: str, map_location: Any = "cpu") -> bool:
    try:
        state, _ = load_checkpoint_payload(path, map_location=map_location)
        model.load_state_dict(state)
        return True
    except Exception:
        return False


def load_model_state(model: torch.nn.Module, path: str, map_location: Any = "cpu") -> Optional[Dict[str, Any]]:
    state, meta = load_checkpoint_payload(path, map_location=map_location)
    model.load_state_dict(state)
    return meta
