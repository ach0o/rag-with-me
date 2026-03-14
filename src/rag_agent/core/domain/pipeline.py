from typing import Any, Protocol, Self

class PipelineStage(Protocol):
    def process(self, input: Any) -> Any:
        ...

class Pipeline:
    def __init__(self, stages: list[PipelineStage] | None = None) -> None:
        self.stages: list[PipelineStage] = stages or []

    def add_stage(self, stage: PipelineStage) -> Self:
        self.stages.append(stage)
        return self

    def run(self, data: Any) -> Any:
        result = data
        for stage in self.stages:
            result = stage.process(result)
        return result
