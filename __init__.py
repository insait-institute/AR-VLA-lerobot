"""ART policy package for LeRobot."""

try:
    import lerobot  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "lerobot is not installed. Please install lerobot to use this policy package."
    ) from exc

from .configuration_art import ARTConfig
from .model_art import ARTPolicy
from .processor_art import make_art_pre_post_processors


def _patch_lerobot_factories() -> None:
    try:
        from lerobot.policies import factory as policy_factory
        from lerobot.scripts import lerobot_train as train_module
    except Exception:
        return

    if getattr(policy_factory.get_policy_class, "_art_patched", False):
        return

    original_get_policy_class = policy_factory.get_policy_class
    original_make_policy_config = policy_factory.make_policy_config
    original_make_pre_post_processors = policy_factory.make_pre_post_processors

    def get_policy_class(name: str):
        if name == "art":
            return ARTPolicy
        return original_get_policy_class(name)

    def make_policy_config(policy_type: str, **kwargs):
        if policy_type == "art":
            return ARTConfig(**kwargs)
        return original_make_policy_config(policy_type, **kwargs)

    def make_pre_post_processors(policy_cfg, pretrained_path=None, **kwargs):
        if isinstance(policy_cfg, ARTConfig):
            return make_art_pre_post_processors(
                config=policy_cfg,
                dataset_stats=kwargs.get("dataset_stats"),
            )
        return original_make_pre_post_processors(
            policy_cfg, pretrained_path=pretrained_path, **kwargs
        )

    get_policy_class._art_patched = True
    policy_factory.get_policy_class = get_policy_class
    policy_factory.make_policy_config = make_policy_config
    policy_factory.make_pre_post_processors = make_pre_post_processors
    train_module.make_pre_post_processors = make_pre_post_processors


_patch_lerobot_factories()

__all__ = [
    "ARTConfig",
    "ARTPolicy",
    "make_art_pre_post_processors",
]
