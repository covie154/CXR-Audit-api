from openai import OpenAI
import json_repair
import json
import pandas as pd
import re

def getLLMJSON(message_lst, client, model_name, response_format):
    """
    Sends a list of messages to a language model (LLM) and attempts to parse the response as JSON.
    Args:
        message_lst (list): A list of messages to send to the LLM.
        client (OpenAI): An instance of the OpenAI client.
        model_name (str): The name of the model to use.
        response_format (BaseModel): Pydantic model for structured output.

    Returns:
        dict: The parsed JSON object from the LLM response.

    Raises:
        ValueError: If no valid JSON object is found in the response after multiple attempts.

    Notes:
        - If the initial response cannot be parsed as JSON, the function will attempt to extract a valid JSON object using regex.
        - If regex extraction fails, the function will ask the LLM for help in debugging the JSON.
        - If all attempts to parse the response as JSON fail, a ValueError is raised.
        - With the new JSON schema format, json_repair shouldn't be necessary anymore.
    """

    try:
        # Use structured outputs with OpenAI API
        completion = client.beta.chat.completions.parse(
            model=model_name,
            messages=message_lst,
            response_format=response_format,
            temperature=0.2
        )
        
        # Extract the parsed content
        resp_json = completion.choices[0].message.parsed
        
        # Convert Pydantic model to dict if needed
        if hasattr(resp_json, 'model_dump'):
            return resp_json.model_dump()
        elif hasattr(resp_json, 'dict'):
            return resp_json.dict()
        else:
            return resp_json
            
    except Exception as e:
        print(f"Structured output failed: {e}")
        # Fallback to regular completion
        completion = client.chat.completions.create(
            model=model_name,
            messages=message_lst,
            temperature=0.2
        )
        
        resp_content = completion.choices[0].message.content
        
        try:
            # Try to parse the response content as JSON
            resp_json = json_repair.loads(resp_content)
            # Return the parsed JSON object
            return resp_json
        except json.JSONDecodeError:
            # If JSON decoding fails, print an error message and the response content
            print(f'JSONDecodeError!')
            print(f'\n\nGot:\n{resp_content}\n\nRetrying...')
            
        # Attempt to find a valid JSON object within the response content using regex
        match = re.search(r'\{.*\}', resp_content, re.DOTALL)
        if match:
            # If a valid JSON object is found, parse it
            intermediate = match.group()
        else:
            # If no valid JSON object is found, print an error message
            print('No valid JSON object found in the response. Retrying...')
            
        # Ask the LLM for help
        debug_messages = [{"role": "user", "content": f"Please fix this JSON: {resp_content}"}]
        debug_completion = client.chat.completions.create(
            model=model_name,
            messages=debug_messages,
            temperature=0.2
        )
        intermediate = debug_completion.choices[0].message.content
            
        try:
            # Try to parse the corrected response content as JSON
            resp_json = json_repair.loads(intermediate)
            # Return the parsed JSON object
            return resp_json
        except json.JSONDecodeError:
            # If JSON decoding fails again, we give up
            print(f'JSONDecodeError Again!')
            print(f'\n\nGot:\n{resp_content}')
            raise ValueError("No valid JSON object found in the response")
    
def loopEvaluate(df, func_iter_item, start_no=0, end_no=-1):
    """
    Evaluates each row of a DataFrame using a provided function and appends the results to the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame to be evaluated.
    func_iter_item (function): A function that takes a row of the DataFrame as input and returns a result.
    start_no (int, optional): The starting index for evaluation. Defaults to 0.
    end_no (int, optional): The ending index for evaluation. Defaults to the end of the DataFrame.

    Returns:
    pd.DataFrame: The original DataFrame concatenated with the evaluation results.
    """

    if end_no == -1:
        end_no = len(df)

    # Create a dictionary to store the evaluation results
    dict_eval = {}

    # Iterate over the rows of the dataframe
    for index, row in df.loc[start_no: end_no].iterrows():
        print(f'[{index+1}/{end_no}]', end=' ')
        result_json = func_iter_item(row)
        print(f"Got response: {result_json}")
        dict_eval[index] = result_json

    # Now we build the df from the list
    final_df = pd.DataFrame.from_dict(dict_eval, orient='index')
    # Concatenate the final_df with the original dataframe
    df = pd.concat([df, final_df], axis=1)
    return df
