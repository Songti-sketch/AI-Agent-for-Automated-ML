import logging
import subprocess

_logger =logging.getLogger(__name__)

def run_command_and_get_output(command):
    try:
        # Run the command and capture the output
        result = subprocess.run(
            command,
            check=True,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10*60, # 10-min
        )
        # Return stdout if successful
        return result.stdout
    except subprocess.CalledProcessError as e:
        # Return stderr if the command fails
        return f"Error: {e.stderr}"

def extract_python_code(text)-> str:
    """
    Extracts Python code from a text string that contains a code block.
    The code block should be enclosed in triple backticks (```).
    The function assumes that the code block is in the format:
    ```python
    # Your code here
    ```
    """
    _logger.debug(f"Model output: \n -----\n{text}\n -----")
    
    if "```python" not in text:
        _logger.warning("No python code block found in the text.")
        print(text)
        return ""
    
    if "```" not in text.split("```python")[1]:
        _logger.warning("No code block found in the text.")
        return ""
    
    # check if there are more than one code blocks
    if text.count("```python") > 2:
        _logger.warning("More than one code block found in the text.")
        return ""

    return text.split("```python")[1].split("```")[0].strip()

def exec_code(code_str: str) -> tuple[int, str]:
    """
    Executes the given Python code string and returns the output as a string.
    """
    # s = os.popen()
    ret = run_command_and_get_output(
        "python3 -c \"{}\"".format(code_str.replace("\"", "\\\""))
    )
    
    # ret = s.read()
    # s.close()
    for idx, err in enumerate([
        "SyntaxError",
        "IndentationError",
        "NameError",
        "TypeError",
        "ValueError",
        "AttributeError",
        "KeyError"
    ]):
        if err in ret:
            return idx+1, ret
    else:
        if "Error" in ret:
            return -1, ret
    return 0, ret