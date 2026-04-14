import importlib

from omegaconf import OmegaConf

from dockstrat.engine import dock_engine  # noqa: F401


def _resolve_omegaconf_variable(variable_path: str):
    """Resolve a dotted module.attribute path to its value."""
    parts = variable_path.rsplit(".", 1)
    module_name = parts[0]
    try:
        module = importlib.import_module(module_name)
        return getattr(module, parts[1])
    except Exception:
        module = importlib.import_module(".".join(module_name.split(".")[:-1]))
        inner = module_name.split(".")[-1]
        return getattr(getattr(module, inner), parts[1])


def register_custom_omegaconf_resolvers():
    """Register custom OmegaConf resolvers used in dockstrat_config YAML files."""
    OmegaConf.register_new_resolver(
        "resolve_variable",
        lambda variable_path: _resolve_omegaconf_variable(variable_path),
    )
