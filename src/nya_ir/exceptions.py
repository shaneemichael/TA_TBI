"""Project-specific exceptions."""


class NyaIRError(Exception):
    """Base class for project exceptions."""


class OptionalDependencyError(NyaIRError, ImportError):
    """Raised when a lazily-loaded optional dependency is missing."""

    def __init__(self, package: str, extra: str | None = None) -> None:
        hint = f" Install with `pip install .[{extra}]`." if extra else ""
        super().__init__(f"Optional dependency `{package}` is required for this operation.{hint}")
        self.package = package
        self.extra = extra

