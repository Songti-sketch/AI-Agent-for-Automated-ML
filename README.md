# NLP Final Project: Agents for Automatic Machine Learning Task Completion

__Authors:__ Zichuan Wan, Jiarui Sun, Xian Ju

### Project Overview
This project aims to develop a agent to automatically complete machine learning tasks using large language models (LLMs). The agent is designed to handle various tasks, including data preprocessing, model selection, hyperparameter tuning, and evaluation. The project leverages the capabilities of LLMs to understand and execute complex machine learning workflows.
The agent is built using the LangChain framework, which provides a modular and flexible architecture for building LLM-based applications. The project also incorporates various tools and libraries for data manipulation, model training, and evaluation.

### Getting Started

To get started with the project, follow these steps:

1. Check your python version:
    ```bash
    python --version
    ```
    The project is developed and tested with Python 3.12. But it should work with Python 3.10 or later.

2. Install the required dependencies:
   ```bash
   pip install -e .
   ```
   The required dependencies are listed in the `pyproject.toml` file.

3. Run a demo:
    ```bash
    cd examples
    DEEPSEEK_API_KEY=<your_deepseek_api_key> python workflow.py
    ```
