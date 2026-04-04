"""
Strategy Registry
─────────────────
A decorator-based registry that maps strategy type strings to their
factory functions.  Eliminates the large if-elif chain in the API layer
and makes it trivial to add new strategies.

Usage:
  from src.simulator.registry import STRATEGY_REGISTRY, get_strategy_fn

  # Register a new strategy (usually in backtester.py):
  @register_strategy("sma_crossover")
  def sma_crossover_strategy(fast_window=50, slow_window=200): ...

  # Look it up later:
  fn = get_strategy_fn("sma_crossover", {"fast_window": 10, "slow_window": 50})
"""

import logging
from typing import Callable, Any

logger = logging.getLogger(__name__)

# ── Registry Storage ─────────────────────────────────────────────────────

# key: strategy type string  →  value: factory function
STRATEGY_REGISTRY: dict[str, Callable] = {}

# Optional metadata (description, param schema) per strategy type
STRATEGY_META: dict[str, dict[str, Any]] = {}


def register_strategy(
    stype: str,
    description: str = "",
    param_schema: dict[str, str] | None = None,
):
    """
    Decorator to register a strategy factory function.

    Parameters
    ----------
    stype : str
        The canonical type key (e.g. ``"sma_crossover"``).
    description : str
        Human-readable description for the API / docs.
    param_schema : dict
        ``{param_name: description_with_default}`` for documentation.
    """

    def _decorator(fn: Callable) -> Callable:
        STRATEGY_REGISTRY[stype] = fn
        STRATEGY_META[stype] = {
            "description": description or fn.__doc__ or "",
            "params": param_schema or {},
        }
        return fn

    return _decorator


def get_strategy_fn(stype: str, params: dict) -> Callable:
    """
    Look up a registered strategy factory and call it with *params*.

    Returns the SignalFunction produced by the factory.

    Raises
    ------
    ValueError
        If *stype* is not registered.
    """
    factory = STRATEGY_REGISTRY.get(stype)
    if factory is None:
        available = ", ".join(sorted(STRATEGY_REGISTRY.keys()))
        raise ValueError(
            f"Unknown strategy type: '{stype}'. "
            f"Available: {available}"
        )
    # Call the factory with the user-supplied params as kwargs
    # Filter to only the params the factory accepts
    import inspect

    sig = inspect.signature(factory)
    valid_keys = set(sig.parameters.keys())
    filtered = {}
    for k, v in params.items():
        if k in valid_keys:
            # Coerce types to match the annotation / default
            default = sig.parameters[k].default
            if default is not inspect.Parameter.empty:
                if isinstance(default, int):
                    v = int(v)
                elif isinstance(default, float):
                    v = float(v)
            filtered[k] = v
    return factory(**filtered)


def list_registered_strategies() -> dict[str, dict]:
    """Return the full registry metadata for all strategies."""
    return dict(STRATEGY_META)
