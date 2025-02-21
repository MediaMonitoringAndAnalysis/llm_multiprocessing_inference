from typing import Dict, List, Literal, Union
import aiohttp
import asyncio
import ssl
import certifi
from tqdm import tqdm
import json
import os
from ast import literal_eval
import re

# from src.utils import _extract_structured_data
from copy import copy

api_pipleines = {
    "OpenAI": {
        "model": "gpt-4o",
        "url": "https://api.openai.com/v1/chat/completions",
        "rate_limit": 5
    },
    "Perplexity": {
        "model": "llama-3.1-sonar-small-128k-chat",
        "url": "https://api.perplexity.ai/chat/completions",
        "rate_limit": 2
    }
}


def _extract_and_evaluate_first(string, default_response):
    start_char = str(default_response)[0]
    end_char = str(default_response)[-1]
    escaped_start = re.escape(start_char)
    escaped_end = re.escape(end_char)

    # Build the regex pattern using the specified start and end characters
    pattern = fr'{escaped_start}.*{escaped_end}'

    # Extract the substring using the regex pattern
    result = re.search(pattern, string)

    # Check if the result was found and return it
    if result:
        return result.group(0)
    else:
        return str(default_response)

def _remove_commas_between_numbers(text):
    # This pattern matches commas between two digits
    pattern = r'(?<=\d),(?=\d)'
    # Replace matched commas with an empty string
    return re.sub(pattern, '', text)

def _posprocess_gpt_output(s, default_response):
    # Remove trailing commas from objects and arrays
    s = re.sub(r",(\s*[}\]])", r"\1", s)
    s = (
        s.replace("```", "")
        .replace("json", "")
        .replace("\n{", "{")
        .replace("}\n", "}")
        .replace("\n", " ")
        .replace("\t", " ")
        .replace("\\xa0", "\\u00A0")
        .strip()
    )
    
    s = _remove_commas_between_numbers(s)
    
    s = _extract_and_evaluate_first(s, default_response)

    return s


# Call ChatGPT with the given prompt, asynchronously.
async def _call_chatgpt_async(
    semaphore: asyncio.Semaphore,
    session,
    message: List[Dict[str, str]],
    response_type: Literal["structured", "unstructured"],
    model: str,
    api_pipeline: Literal["OpenAI", "Perplexity"],
    default_response: str,
    api_key: str,
):
    
    final_model = api_pipleines[api_pipeline]["model"] if model is None else model
    url = api_pipleines[api_pipeline]["url"]
    
    payload = {
        "model": final_model,
        "messages": message,
        "temperature": 0.0,
    }
    
    async with semaphore:
        async with session.post(
            url=url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json=payload,
            ssl=ssl.create_default_context(cafile=certifi.where()),
        ) as response:
            clean_response = await response.json()
            # st.markdown(clean_response)

        try:
            output_text = clean_response["choices"][0]["message"]["content"]
            # st.markdown(output_text)
            # print("Worked")
        except Exception as e:
            print("GPT running failed", e, clean_response)
            output_text = str(default_response)

    output_text = output_text.replace("json\n", "")
    if response_type == "structured":
        output_text = _posprocess_gpt_output(output_text, default_response)
        try:
            gpt_extracted_infos = literal_eval(output_text)
        except:
            print("literal_eval failed", output_text)
            try:
                gpt_extracted_infos = json.loads(output_text)
            except Exception as e:
                # st.markdown("Error in extracting structured data", e)
                # st.markdown(output_text)
                gpt_extracted_infos = default_response
    else:
        gpt_extracted_infos = output_text

    return gpt_extracted_infos


# Call chatGPT for all the given prompts in parallel.
async def _call_chatgpt_bulk(
    messages: List[List[Dict[str, str]]],
    default_response: str,
    response_type: Literal["structured", "unstructured"],
    model: str,
    api_key: str,
    api_pipeline: Literal["OpenAI", "Perplexity"],
    specific_description: str = "",
):
    final_description = "Processing Data"
    if specific_description != "":
        final_description += f": {specific_description}"
    rate_limit = api_pipleines[api_pipeline]["rate_limit"]
    assert response_type in ["structured", "unstructured"]
    # print("Messages", messages)
    semaphore = asyncio.Semaphore(rate_limit)  # Control concurrency level
    async with aiohttp.ClientSession() as session:
        # Create a tqdm async progress bar
        progress_bar = tqdm(total=len(messages), desc=final_description, position=0)

        async def wrapped_call(session, message):
            # Wrap your call in a function that updates the progress bar
            try:
                return await _call_chatgpt_async(
                    semaphore,
                    session,
                    message=message,
                    response_type=response_type,
                    model=model,
                    default_response=default_response,
                    api_key=api_key,
                    api_pipeline=api_pipeline,
                )
            finally:
                progress_bar.update(1)  # Update the progress for each completed task

        # Use asyncio.TaskGroup for managing tasks
        async with asyncio.TaskGroup() as tg:
            tasks = [
                tg.create_task(wrapped_call(session, message)) for message in messages
            ]
            responses = await asyncio.gather(*tasks)

        # Ensure the progress bar closes properly
        progress_bar.close()

    return responses


def get_answers(
    prompts: List[List[Dict[str, str]]],
    default_response: Union[str, List, Dict],
    response_type: Literal["structured", "unstructured"],
    api_pipeline: Literal["OpenAI", "Perplexity"],
    api_key: str,
    model: str = None,
    specific_description: str = "",
) -> List[Union[str, Dict[str, Union[str, float]]]]:

    try:
        answers: Union[str, Dict[str, Union[str, float]]] = asyncio.run(
            _call_chatgpt_bulk(
                messages=prompts,
                response_type=response_type,
                model=model,
                default_response=str(default_response),
                api_key=api_key,
                api_pipeline=api_pipeline,
                specific_description=specific_description,
            )  # _call_chatgpt_bulk(prompts, "{}", "structured")
        )
    except ExceptionGroup as e:
        print(f"Caught an ExceptionGroup with {len(e.exceptions)} sub-exceptions:")
        for exception in e.exceptions:
            print(f"- {type(exception).__name__}: {exception}")
        answers = [default_response for _ in range(len(prompts))]

    return answers
