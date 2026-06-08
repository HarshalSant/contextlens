"""Pricing table and cost calculation utilities."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    """Per-token pricing for a model (USD)."""

    input_per_million: float
    output_per_million: float

    def input_cost(self, tokens: int) -> float:
        return tokens * self.input_per_million / 1_000_000

    def output_cost(self, tokens: int) -> float:
        return tokens * self.output_per_million / 1_000_000


# Prices as of mid-2025; override via CostModel if stale.
_DEFAULT_PRICING: dict[str, ModelPricing] = {
    # Anthropic
    "claude-opus-4-8": ModelPricing(15.0, 75.0),
    "claude-opus-4-7": ModelPricing(15.0, 75.0),
    "claude-opus-4-6": ModelPricing(15.0, 75.0),
    "claude-opus-4-5": ModelPricing(15.0, 75.0),
    "claude-sonnet-4-6": ModelPricing(3.0, 15.0),
    "claude-sonnet-4-5": ModelPricing(3.0, 15.0),
    "claude-haiku-4-5": ModelPricing(0.8, 4.0),
    "claude-3-5-sonnet-20241022": ModelPricing(3.0, 15.0),
    "claude-3-5-haiku-20241022": ModelPricing(0.8, 4.0),
    "claude-3-opus-20240229": ModelPricing(15.0, 75.0),
    "claude-3-sonnet-20240229": ModelPricing(3.0, 15.0),
    "claude-3-haiku-20240307": ModelPricing(0.25, 1.25),
    # OpenAI
    "gpt-4o": ModelPricing(2.5, 10.0),
    "gpt-4o-mini": ModelPricing(0.15, 0.6),
    "gpt-4-turbo": ModelPricing(10.0, 30.0),
    "gpt-4": ModelPricing(30.0, 60.0),
    "gpt-3.5-turbo": ModelPricing(0.5, 1.5),
    "o1": ModelPricing(15.0, 60.0),
    "o1-mini": ModelPricing(3.0, 12.0),
    "o3-mini": ModelPricing(1.1, 4.4),
}

_FALLBACK_PRICING = ModelPricing(3.0, 15.0)


class CostModel:
    """Holds the pricing table; can be customized per-project."""

    def __init__(self, overrides: dict[str, ModelPricing] | None = None) -> None:
        self._table: dict[str, ModelPricing] = dict(_DEFAULT_PRICING)
        if overrides:
            self._table.update(overrides)

    def pricing_for(self, model: str) -> ModelPricing:
        if model in self._table:
            return self._table[model]
        # Fuzzy match: strip version suffixes and try prefix match
        lower = model.lower()
        for key in self._table:
            if lower.startswith(key.lower()) or key.lower().startswith(lower):
                return self._table[key]
        return _FALLBACK_PRICING

    def input_cost(self, model: str, tokens: int) -> float:
        return self.pricing_for(model).input_cost(tokens)

    def output_cost(self, model: str, tokens: int) -> float:
        return self.pricing_for(model).output_cost(tokens)

    def register(self, model: str, pricing: ModelPricing) -> None:
        self._table[model] = pricing

    @property
    def known_models(self) -> list[str]:
        return sorted(self._table.keys())


# Module-level default instance
default_cost_model = CostModel()
