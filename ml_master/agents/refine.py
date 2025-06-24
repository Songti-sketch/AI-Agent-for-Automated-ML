import os
import time
# from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import random
import json
# from retrying import retry
import requests

from ..misc import parse_prob_to_desc, extract_python_code


_refiner_prompt = """
You are an expert in the field of AI and ML. You are now given piece of code that implements a piece of python code
that completes a ML task, and it's execution result. Based on the execution result, you need to refine the code to make
it better.

Please follow these steps to complete the task:
1. Is the code runnable?
2. If the code is not runnable, what can we do to make it work?
3. If the code is runnable, interprete the result, and think: can we further improve the performance?
4. If you believe you could improve the performance, output the modified code.
5. If the performance is already good to go, don't output any python code.
6. If, based on the result, you believe the ML model is not suitable for this task, do not output any python code.

The format of the Python code should be inclosed in a code block.:
```python
# Your code here
```

There should be at most one python code block in the output.
The code should be complete and runnable.
"""

def _parse_prob_to_desc_prompt(df, target_column: str, metric_name: str, code_str, exec_result) -> str:
    """
    Parse the dataframe to a string description.
    """
    return parse_prob_to_desc(df, target_column, metric_name) + """
Addtional requirements:
- Is the ML model even suitable for this task? If it't not suitable, give us reason and DO NOT output any python code.
- The code block must be explicitly marked as a python code block. That is, the code block should start with ```python and end with ```.
- DO NOT modify the ML method. If it is done with a method (e.g. neural network), please still use that method.
- The code should be complete and runnable.
- You need to report the accuracy of the model.
- Think of the way we preprocess the data. Is the original implementatiopn correct?
- If there are error message, think how to modify the code such that at least it is runnable.
- use multiple hyperparameters and find the best one.
- We have limited computation resource. You could do grid search but don't be ridiculous (e.g. having more than 100 combination of params.)
""" + f"""
The current python code implementation: 
```python
{code_str}
```
""" + f"""
The execution result:
{exec_result}
"""

refiner_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _refiner_prompt),
        ("user", "{input}")
    ]
)

model = ChatDeepSeek(model="deepseek-chat")

refiner = _parse_prob_to_desc_prompt | refiner_prompt | model | StrOutputParser() | extract_python_code

# --- judge ---

_judge_prompt = """
You are an exper in the field of AI and ML. There are currectly several pieces of python code
that completes a ML task. Based on the execution result, judge which one is better.

Please just ouput which is better(the result index) and DO NOT output anything else.
"""

def _parse_exec_results_to_desc_prompt(
    exec_results: list
):
    ret = ""
    for idx, exec_result in enumerate(exec_results):
        ret += f"""

--- result idx={idx} ---
```
{exec_result}
```
"""
    return ret

def _get_index(ret_str: str):
    ret_str = ret_str.split("/n")
    return int(ret_str[-1])

judge_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _judge_prompt),
        ("user", "{input}")
    ]
)

model = ChatDeepSeek(model="deepseek-chat")

judge = _parse_exec_results_to_desc_prompt | judge_prompt | model | StrOutputParser() | _get_index