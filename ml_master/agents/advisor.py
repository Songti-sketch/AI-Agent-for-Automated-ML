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

from ..misc import parse_prob_to_desc

_advisor_prompt = """
You are an expert in the field of AI and ML. You are now given a complex data analysis task.
To complete the task, you need to give me advice of what ML model to use. 

Please follow these steps to complete the task:

1. Analyze the data provided in the input using pandas and numpy.
2. Based on the analysis, review what ML models are sutiable for the task. Available models are:
    - Random Forest
    - Support Vector Machine
    - Neural Network
    - Logistic Regression
    - Linear Regression
    - Naive Bayes
3. Provide a brief explanation of why you chose the model.
4. Ouput the final result. It should follow this format:
<model_name1> <model_name2> ... <model_nameN>

---- Sample Output ----
From the data provided, I can see that the data is a classification task with a binary target variable. The features are continuous and categorical. 

The random forest model is also a good choice as it can handle overfitting and is robust to noise.
The support vector machine model is also a good choice as it can handle high dimensional data and is robust to noise.
The linear regression model is not a good choice as it assumes a linear relationship between the features and the target variable.
The neural network model is not a good choice as it requires a lot of data and is prone to overfitting.

Based on this analysis, I would recommend using the following models:

<Random Forest> <Neural Network> <Support Vector Machine>
"""

def _parse_prob_to_desc_prompt(df, target_column: str, metric_name: str) -> str:
    """
    Parse the dataframe to a string description.
    """
    return parse_prob_to_desc(df, target_column, metric_name) + (
        f"Please analyze the data and provide a recommendation of what ML model to use.\n"
        f"Again note that we should keep the recommendation format correct, that is, the last line should look like:\n"
        f"<model_name1> <model_name2> ... <model_nameN>\n"
    )

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _advisor_prompt),
        ("user", "{input}")
    ]
)

model = ChatDeepSeek(model="deepseek-chat")

def parse_recommendation(ai_message: AIMessage) -> list[str]:
    """
    Parse the recommendation from the AI message.
    The recommendation is always in the last line.
    """
    # Split the message into lines
    lines = ai_message.content.split("\n")
    # Get the last line
    recommendation = lines[-1]
    # Remove any leading or trailing whitespace
    recommendation = recommendation.strip()
    
    # Split the recommendation into a list of models based on `<>`
    ret = []
    for model in recommendation.split("<"):
        # Remove any leading or trailing whitespace
        model = model.strip()
        # Remove any trailing `>`
        if ">" in model:
            model = model.split(">", maxsplit=1)[0]
        # Add the model to the list
        if model:
            ret.append(model.strip())
        
    # validate the models
    valid_models = [
        "Random Forest",
        "Support Vector Machine",
        "Neural Network",
        "Logistic Regression",
        "Linear Regression",
        "Naive Bayes"
    ]
    
    
    for model in ret:
        if model not in valid_models:
            print(f"[WARNING] Invalid model: {model}. Valid models are: {valid_models}")
            
    # Return the recommendation
    return [model for model in ret if model in valid_models]

    
model_advisor = _parse_prob_to_desc_prompt | prompt | model | parse_recommendation
model_advisor.__doc__ = (
    """
    Given a pandas dataframe, return a list of ML models that are suitable for the task.
    """
)
