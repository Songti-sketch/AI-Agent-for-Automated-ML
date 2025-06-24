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


_implementor_prompt = """
You are an expert in the field of AI and ML. You are now given a complex data analysis task.
To complete the task, you need to use Linear Regression to implement the task.

Please follow these steps to complete the task:
1. Analyze the data provided in the input using pandas and numpy.
2. Based on the analysis, implement the task using Linear Regression.

The format of the implementation should be in Python code. It should be inclosed in a code block.:
```python
# Your code here
```

There should be one and only one python code block in the output.
The code should be complete and runnable.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _implementor_prompt),
        ("user", "{input}")
    ]
)

model = ChatDeepSeek(model="deepseek-chat")

def _parse_prob_to_desc_prompt(df, data_file_name, target_column: str, metric_name: str) -> str:
    """
    Parse the dataframe to a string description.
    """
    return parse_prob_to_desc(df, target_column, metric_name) + f"""
Addtional requirements:
- The code block must be explicitly marked as a python code block. That is, the code block should start with ```python and end with ```.
- There should be one and only one python code block in the output.
- DO NOT hard-code the data frame, it is a csv file in directory `{data_file_name}`.
- The code should be complete and runnable.
- You need to report the accuracy of the model.
- Think of what column is not needed and drop it. Of course, you need to explain why you drop it in comments.
- Think of how to test the model. Train-test split? N-fold cross validation?
- use multiple hyperparameters and find the best one.
- When using PyTorch, try to use GPU when possible.
"""

lr_implementor = _parse_prob_to_desc_prompt | prompt | model | StrOutputParser() | extract_python_code
lr_implementor.__doc__ = (
    """
    This function takes a dataframe, target column, and metric name as input and returns a python code string that implements the task using Random Forest.
    The function uses the ChatDeepSeek model to generate the code.
    The code is extracted from the output of the model using the extract_python_code function.
    """
)
