from rag_agent.domain.pipeline import Pipeline


class AddOne:
    def process(self, data: int) -> int:
        return data + 1


class Double:
    def process(self, data: int) -> int:
        return data * 2


def test_empty_pipeline_returns_input():
    # Given: an empty pipeline with no stages
    pipeline = Pipeline()

    # When: we run it with input 42
    result = pipeline.run(42)

    # Then: it should return the input unchanged
    assert result == 42


def test_single_stage():
    # Given: a pipeline with one AddOne stage
    pipeline = Pipeline().add_stage(AddOne())

    # When: we run it with 0
    result = pipeline.run(0)

    # Then: we get 1
    assert result == 1


def test_stages_run_in_order():
    # Given: a pipeline with AddOne then Double
    pipeline = Pipeline().add_stage(AddOne()).add_stage(Double())

    # When: we run it with 3
    result = pipeline.run(3)

    # Then: it computes (3 + 1) * 2 = 8, not (3 * 2) + 1 = 7
    assert result == 8


def test_fluent_chaining():
    # Given: three AddOne stages chained fluently
    pipeline = Pipeline().add_stage(AddOne()).add_stage(AddOne()).add_stage(AddOne())

    # When: we run it with 0
    result = pipeline.run(0)

    # Then: each stage adds 1, giving 3
    assert result == 3
