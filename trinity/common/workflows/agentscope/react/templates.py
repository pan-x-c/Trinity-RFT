from dataclasses import dataclass
from typing import Dict, Optional, Type

from pydantic import BaseModel, Field

from trinity.common.rewards import MathBoxedRewardFn, RewardFn

# For GSM8K task


class GSM8KStructure(BaseModel):
    response: str


GSM8KSystemPrompt = """You are an agent specialized in solving math problems with tools. Please solve the math problem given to you. You can write and execute Python code to perform calculation or verify your answer. You should return your final answer within \\boxed{{}}."""


class GSM8KRewardFn(MathBoxedRewardFn):
    def __call__(  # type: ignore [override]
        self,
        response: str,
        truth: str,
        format_score_coef: Optional[float] = 0.1,
        **kwargs,
    ) -> dict[str, float]:
        # parse GSM8K truth
        if isinstance(truth, str) and "####" in truth:
            truth = truth.split("####")[1].strip()
        else:
            truth = str(truth)
        return super().__call__(
            response=response,
            truth=truth,
            with_think=False,
            format_score_coef=format_score_coef,
            **kwargs,
        )


class GSM8KResponseStructure(BaseModel):
    result: str = Field(
        description="Your solution of the given math problem. Put your final answer in boxed format, e.g., \\boxed{42}"
    )


# Registry for different templates


@dataclass
class Template:
    """A template for different task types, including system prompt and response structure."""

    system_prompt: str
    response_structure: BaseModel
    reward_fn_cls: Type[RewardFn]


TEMPLATE_MAP: Dict[str, Optional[Template]] = {
    "gsm8k": Template(
        system_prompt=GSM8KSystemPrompt,
        response_structure=GSM8KResponseStructure,
        reward_fn_cls=GSM8KRewardFn,
    ),
    # Add more templates for different task types as needed
}
