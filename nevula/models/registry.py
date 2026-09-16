from typing import Any, Callable, Dict, List, Optional, Type, Union

# Global registry storing model metadata
# Key: canonical name (str), Value: dict(class=ModelClass, category=str, display_name=str)
_REGISTRY: Dict[str, Dict[str, Any]] = {}


def _normalize_name(name: str) -> str:
    """Converts a model name to canonical snake_case lowercase representation."""
    import re
    # Handle CamelCase to snake_case
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.lower().replace("-", "_").strip()


def register_model(
    model_or_name: Union[Type[Any], str, None] = None,
    name: Optional[str] = None,
    category: str = "general",
) -> Any:
    """
    Decorator or function to register a model class into the Nevula Model Library.

    Usage as decorator:
        @register_model(name="linear_regression", category="regression")
        class LinearRegression(BaseModel):
            ...

        @register_model
        class LogisticRegression(BaseModel):
            ...

    Usage as function:
        register_model(LinearRegression, name="linear_regression", category="regression")
    """
    model_name = name or (model_or_name if isinstance(model_or_name, str) else None)

    def _decorator(cls: Type[Any]) -> Type[Any]:
        raw_name = model_name or cls.__name__
        canon_name = _normalize_name(raw_name)
        _REGISTRY[canon_name] = {
            "class": cls,
            "category": category.lower(),
            "display_name": cls.__name__,
            "canonical_name": canon_name,
        }
        return cls

    if isinstance(model_or_name, type):
        # Called as @register_model directly on class
        return _decorator(model_or_name)
    else:
        return _decorator


def get_model(name: str) -> Type[Any]:
    """
    Retrieves a registered model class by name.

    Args:
        name: Name of the model (case-insensitive, snake_case or CamelCase).

    Returns:
        The model class.

    Raises:
        KeyError: If model is not registered.
    """
    canon = _normalize_name(name)
    if canon not in _REGISTRY:
        available = sorted(list(_REGISTRY.keys()))
        raise KeyError(
            f"Model '{name}' not found in registry. "
            f"Available registered models: {available}."
        )
    return _REGISTRY[canon]["class"]


def list_models(category: Optional[str] = None) -> Union[List[str], Dict[str, List[str]]]:
    """
    Lists available models in the registry.

    Args:
        category: Optional category filter (e.g. 'regression', 'classification', 'deep').

    Returns:
        If category is provided, returns list of model display names in that category.
        If category is None, returns dict mapping category -> list of model display names.
    """
    if category is not None:
        cat_lower = category.lower().strip()
        names = {
            meta["display_name"]
            for meta in _REGISTRY.values()
            if meta["category"] == cat_lower
        }
        return sorted(list(names))

    categorized: Dict[str, set] = {}
    for meta in _REGISTRY.values():
        cat = meta["category"].capitalize()
        categorized.setdefault(cat, set()).add(meta["display_name"])

    return {cat: sorted(list(names)) for cat, names in categorized.items()}


def summary() -> str:
    """
    Returns a human-readable formatted summary of all available models.
    """
    catalog = list_models()
    if not catalog:
        return "No models currently registered in Nevula Model Library."

    lines = ["Available models in Nevula Model Library:\n"]
    for cat, models in sorted(catalog.items()):
        lines.append(f"{cat}:")
        for m in models:
            lines.append(f"  - {m}")
        lines.append("")
    return "\n".join(lines)


__all__ = [
    "register_model",
    "get_model",
    "list_models",
    "summary",
]
