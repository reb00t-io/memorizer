from memorizer.model.context import Context
from memorizer.model.model import DEFAULT_GOAL_PLACEHOLDER, Model


def test_model_init_sets_default_goal_when_missing() -> None:
    ctx = Context.create(system_prompt="You are <MODEL_ID>", persist=False)
    model = Model(ctx, model_id="test-model", model_name="t", max_completion_tokens=128)

    assert model.context.model_goal.messages()[0].content == DEFAULT_GOAL_PLACEHOLDER


def test_model_init_keeps_existing_goal() -> None:
    ctx = Context.create(system_prompt="You are <MODEL_ID>", persist=False)
    ctx.model_goal.append("memory", "Keep this goal")

    Model(ctx, model_id="test-model", model_name="t", max_completion_tokens=128)

    assert ctx.model_goal.messages()[0].content == "Keep this goal"


def test_model_create_builds_context_and_config() -> None:
    model = Model.create(
        model_id="test-model",
        system_prompt="You are <MODEL_ID>",
        base_url="http://example.test/v1",
        max_completion_tokens=256,
        persist=False,
    )

    assert model.model_id == "test-model"
    assert model.model_name == "test-model"
    assert model.base_url == "http://example.test/v1"
    assert model.max_completion_tokens == 256
    # MODEL_ID template var resolved in the system prompt.
    assert model.context.system.to_messages()[0]["content"] == "You are test-model"
