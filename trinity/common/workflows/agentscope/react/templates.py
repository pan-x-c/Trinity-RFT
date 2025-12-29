import re
from dataclasses import dataclass
from typing import Dict, Optional, Type

from pydantic import BaseModel, Field

from agentscope.message import Msg

from trinity.common.rewards import RewardFn
from trinity.common.rewards.math_reward import MathBoxedRewardFn

# For GSM8K task
GSM8KSystemPrompt = """You are an agent specialized in solving math problems with tools. Please solve the math problem given to you. You can write and execute Python code to perform calculation or verify your answer. You should return your final answer within \\boxed{{}}."""


class GSM8KResponseStructure(BaseModel):
    result: str = Field(
        description="Your solution of the given math problem. Put your final answer in boxed format, e.g., \\boxed{42}"
    )


class GSM8KRewardFn(MathBoxedRewardFn):
    def __call__(  # type: ignore [override]
        self,
        response: Msg,
        truth: str,
        format_score_coef: float = 0.1,
        **kwargs,
    ) -> Dict[str, float]:
        # parse GSM8K truth
        if isinstance(truth, str) and "####" in truth:
            truth = truth.split("####")[1].strip()
        else:
            truth = str(truth)
        # parse the final answer from the response message
        result = response.get_text_content()
        if result is not None:
            # find the final answer in boxed format
            match = re.search(pattern=r"\\boxed\{([^}]*)\}", string=result)
            if match:
                result = match.group(1).strip()
            else:
                result = None
        return super().__call__(
            response=result,
            truth=truth,
            with_think=False,
            format_score_coef=format_score_coef,
            **kwargs,
        )


# Registry for different templates


@dataclass
class Template:
    """A template for different task types, including system prompt and response structure."""

    system_prompt: str
    response_structure: Type[BaseModel]
    reward_fn_cls: Type[RewardFn]


TEMPLATE_MAP: Dict[str, Optional[Template]] = {
    "gsm8k": Template(
        system_prompt=GSM8KSystemPrompt,
        response_structure=GSM8KResponseStructure,
        reward_fn_cls=GSM8KRewardFn,
    ),
    # Add more templates for different task types as needed
}
