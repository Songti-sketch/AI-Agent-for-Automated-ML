from ..misc import parse_prob_to_desc

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
"""