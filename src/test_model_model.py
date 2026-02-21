from src.model.context import Context
from src.model.model import DEFAULT_GOAL_PLACEHOLDER, Model


def test_model_init_sets_default_goal_when_missing() -> None:
    ctx = Context.create(system_prompt="You are <MODEL_ID>", persist=False)
    model = Model(ctx, model_info={"model_id": "test-model", "model_name": "t", "system_prompt": "s"})

    assert model.context.model_goal.messages()[0].content == DEFAULT_GOAL_PLACEHOLDER


def test_model_init_keeps_existing_goal() -> None:
    ctx = Context.create(system_prompt="You are <MODEL_ID>", persist=False)
    ctx.model_goal.append("memory", "Keep this goal")

    Model(ctx, model_info={"model_id": "test-model", "model_name": "t", "system_prompt": "s"})

    assert ctx.model_goal.messages()[0].content == "Keep this goal"
