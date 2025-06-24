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


_ensembler_prompt = """
You are an expert in the field of AI and ML. You are now given several ML model implementation
that completes the task. Based on these code, you need to ensemble them all together to make the performance better.

Please follow these steps to complete the task:
1. Check all implementations and the execution results and choose what to emsemble.
2. Implement ensemble algorithm using these models.

The format of the implementation should be in Python code. It should be inclosed in a code block.:
```python
# Your code here
```

There should be one and only one python code block in the output.
The code should be complete and runnable.
"""

def _parse_prob_to_prompt(df, data_file_name, target_column: str, metric_name: str, 
                               all_codes: list[str], execution_results: list[str]) -> str:
    """
    Parse the dataframe to a string description.
    """
    
    prompt = parse_prob_to_desc(df, target_column, metric_name) + f"""
Addtional requirements:
- The code block must be explicitly marked as a python code block. That is, the code block should start with ```python and end with ```.
- There should be one and only one python code block in the output.
- DO NOT hard-code the data frame, it is a csv file in directory f`{data_file_name}`.
- The code should be complete and runnable.
- You need to report the accuracy of the model.
- Think of what column is not needed and drop it. Of course, you need to explain why you drop it in comments.
- Think of how to test the model. Train-test split? N-fold cross validation?
""" + """
--- implementations ---
"""

    for code, exec_result in zip(all_codes, execution_results):
        prompt += f"""
--- code ---
```python
{code}
```
--- result ---
```text
{exec_result}
```
"""
    return prompt

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _ensembler_prompt),
        ("user", "{input}")
    ]
)

model = ChatDeepSeek(model="deepseek-chat")
emsembler = _parse_prob_to_prompt | prompt | model | StrOutputParser() | extract_python_code