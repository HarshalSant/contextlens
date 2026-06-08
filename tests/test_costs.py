"""Tests for the cost model."""

from contextlens.costs import CostModel, ModelPricing, default_cost_model


def test_known_model_pricing() -> None:
    p = default_cost_model.pricing_for("claude-3-5-sonnet-20241022")
    assert p.input_per_million == 3.0
    assert p.output_per_million == 15.0


def test_input_cost_math() -> None:
    p = ModelPricing(input_per_million=3.0, output_per_million=15.0)
    # 1 million tokens = $3.00
    assert p.input_cost(1_000_000) == 3.0
    # 100k tokens = $0.30
    assert abs(p.input_cost(100_000) - 0.30) < 1e-9


def test_output_cost_math() -> None:
    p = ModelPricing(input_per_million=3.0, output_per_million=15.0)
    assert abs(p.output_cost(1_000_000) - 15.0) < 1e-9


def test_custom_override() -> None:
    custom = CostModel(overrides={"my-model": ModelPricing(1.0, 5.0)})
    p = custom.pricing_for("my-model")
    assert p.input_per_million == 1.0


def test_fallback_pricing_for_unknown() -> None:
    p = default_cost_model.pricing_for("totally-unknown-model-xyz")
    # Fallback is $3/$15 — should not raise
    assert p.input_per_million > 0
    assert p.output_per_million > 0


def test_gpt4o_pricing() -> None:
    p = default_cost_model.pricing_for("gpt-4o")
    assert p.input_per_million == 2.5
    assert p.output_per_million == 10.0


def test_register_new_model() -> None:
    cm = CostModel()
    cm.register("acme-llm-v1", ModelPricing(0.5, 2.0))
    assert "acme-llm-v1" in cm.known_models
    assert cm.input_cost("acme-llm-v1", 1_000_000) == 0.5


def test_zero_tokens() -> None:
    assert default_cost_model.input_cost("gpt-4o", 0) == 0.0
