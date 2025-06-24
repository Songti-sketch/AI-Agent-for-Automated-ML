
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

from .common import _parse_prob_to_desc_prompt
from ..misc import extract_python_code

_implementor_prompt = """
You are an expert in the field of AI and ML. You are now given a complex data analysis task.
To complete the task, you need to use Naive Bayes to implement the task.

Please follow these steps to complete the task:
1. Analyze the data provided in the input using pandas and numpy.
2. Based on the analysis, implement the task using Naive Bayes.

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

naive_bayes_implementor = _parse_prob_to_desc_prompt | prompt | model | StrOutputParser() | extract_python_code
naive_bayes_implementor.__doc__ = (
    """
    This function takes a dataframe, target column, and metric name as input and returns a python code string that implements the task using Random Forest.
    The function uses the ChatDeepSeek model to generate the code.
    The code is extracted from the output of the model using the extract_python_code function.
    """
)
