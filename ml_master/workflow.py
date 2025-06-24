import logging

_logger = logging.getLogger(__name__)


def solve(data_file_name, target_col, metric_name="R2"):
    """
    Gives solution to ML problem in python code.
    """
    import pandas as pd
    
    # write dependencies within function to make it private
    from .agents import (
        model_advisor, 
        decision_tree_implementor, 
        logistic_implementor, 
        lr_implementor, 
        naive_bayes_implementor, 
        nn_implementor, 
        svm_implementor,
        refiner,
        judge,
        emsembler
    )
    from .misc import exec_code
    
    recommendation_mapping = {
        "Random Forest": decision_tree_implementor,
        "Support Vector Machine": svm_implementor,
        "Neural Network": nn_implementor,
        "Logistic Regression": logistic_implementor,
        "Linear Regression": lr_implementor,
        "Naive Bayes": naive_bayes_implementor,
    }
    
    recommendation_count = {
        "Random Forest": 0,
        "Support Vector Machine": 0,
        "Neural Network": 0,
        "Logistic Regression": 0,
        "Linear Regression": 0,
        "Naive Bayes": 0,
    }
    
    df = pd.read_csv(data_file_name)
    
    # --- actual workflow ---
    recommendations = []
    for i in range(5):
        recommendations += model_advisor.invoke(
            df, 
            target_column=target_col, # predicting the price of the car
            metric_name="R2", # using R2 to eval
        )
    _logger.info(f"Advisor recommendataion: {recommendations}")
    
    # make recommendation count
    for recommendation in recommendations:
        recommendation_count[recommendation] += 1
    real_recommendations = set(k for k,v in recommendation_count.items() if v>2)
    if len(real_recommendations) == 0:
        real_recommendations = set(k for k,v in recommendation_count.items() if v>1)
    if len(real_recommendations) == 0:
        # still zero...?
        real_recommendations = {"Random Forest"}
    _logger.info(f"Real recommendataion: {real_recommendations}")
    
    all_codes = []
    all_ret = []
    for recommendation in real_recommendations:
        implementor = recommendation_mapping[recommendation]
        code = implementor.invoke(
            df, 
            data_file_name=data_file_name,
            target_column=target_col, 
            metric_name=metric_name,
        )
        _logger.info(f"{recommendation} implemented. Executing...")
        
        ret_code, code_ret_str = exec_code(code)
        _logger.info(f"Execution done with ret_code={ret_code}. Trying to refine..")
        refined_code = refiner.invoke(
            df, 
            target_column=target_col, # predicting the price of the car
            metric_name="R2", # using R2 to eval
            code_str=code,
            exec_result=code_ret_str,
        )
        _logger.info(f"refined code generated. Executing...")
        
        if refined_code != "":
            ret_code2, code_ret_str_v2 = exec_code(refined_code)
        _logger.info(f"Execution done with ret_code={ret_code2} and terminal_output: {code_ret_str_v2}. Judging..")
        if ret_code != 0 and ret_code2 != 0:
            continue # robust to implementation errors
        elif ret_code != 0:
            better_idx = 1
        elif ret_code2 != 0:
            better_idx = 0
        else:
            better_idx = judge.invoke(
                [code_ret_str, code_ret_str_v2]
            )
            if better_idx not in [0, 1]:
                better_idx = 0

        _logger.info(f"Code with idx={better_idx} is better.")
        all_codes.append(
            code if better_idx == 0 else refined_code
        )
        all_ret.append(
            code_ret_str if better_idx == 0 else code_ret_str_v2
        )

    if len(all_codes) > 1:
        ensemble_code = emsembler.invoke(
            df, 
            data_file_name=data_file_name,
            target_column=target_col, 
            metric_name=metric_name,
            all_codes=all_codes,
            execution_results=all_ret,
        )
        ret_code, code_ret_str = exec_code(ensemble_code)
        
        for i in range(3):
            if ret_code == 0:
                break
            _logger.info(f"Code error: {code_ret_str}")
            ensemble_code = refiner.invoke(
                df, 
                target_column=target_col, # predicting the price of the car
                metric_name="R2", # using R2 to eval
                code_str=ensemble_code,
                exec_result=code_ret_str,
            )
            ret_code, code_ret_str = exec_code(ensemble_code)
    else:
        ensemble_code = all_codes[0]
        code_ret_str = code_ret_str if better_idx == 0 else code_ret_str_v2
    
    return ensemble_code, code_ret_str