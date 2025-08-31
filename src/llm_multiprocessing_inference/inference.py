from typing import Dict, List, Literal, Union, Optional, Callable
import aiohttp
import asyncio
import ssl
import certifi
from tqdm import tqdm
import json
import os
from ast import literal_eval
import re
import ollama
import base64
from copy import copy
from Levenshtein import distance as levenshtein

general_pipelines = {
    "OpenAI": {
        "model": "gpt-4o-mini",
        "url": "https://api.openai.com/v1/chat/completions",
        "rate_limit": 5,
    },
    "Perplexity": {
        "model": "llama-3.1-sonar-small-128k-chat",
        "url": "https://api.perplexity.ai/chat/completions",
        "rate_limit": 2,
    },
    "Ollama": {
        "model": "gemma3:4b-it-q4_K_M",
        "url": None,
        "rate_limit": None,
    },
}


def _extract_and_evaluate_first(string, default_response):
    start_char = str(default_response)[0]
    end_char = str(default_response)[-1]
    escaped_start = re.escape(start_char)
    escaped_end = re.escape(end_char)

    # Build the regex pattern using the specified start and end characters
    pattern = rf"{escaped_start}.*{escaped_end}"

    # Extract the substring using the regex pattern
    result = re.search(pattern, string)

    # Check if the result was found and return it
    if result:
        return result.group(0)
    else:
        return str(default_response)


def _remove_commas_between_numbers(text):
    # This pattern matches commas between two digits
    pattern = r"(?<=\d),(?=\d)"
    # Replace matched commas with an empty string
    return re.sub(pattern, "", text)


def replace_unneeded_characters(s):
    return (
        s.replace("```", "")
        .replace("json\n", "")
        .replace("json", "")
        .replace("\n{", "{")
        .replace("}\n", "}")
        .replace("\n", " ")
        .replace("\t", " ")
        .replace("\\xa0", "\\u00A0")
        # .strip()
    )


def _posprocess_gpt_output(s, default_response):
    # Remove trailing commas from objects and arrays
    s = re.sub(r",(\s*[}\]])", r"\1", s).strip()
    s = replace_unneeded_characters(s)
    s = _remove_commas_between_numbers(s)

    s = _extract_and_evaluate_first(s, default_response)

    return s


def postprocess_structured_output(output_text: str, default_response: str):
    output_text = _posprocess_gpt_output(output_text, default_response)
    try:
        gpt_extracted_infos = literal_eval(output_text)
    except:
        print("literal_eval failed", output_text)
        try:
            gpt_extracted_infos = json.loads(output_text)
        except Exception as e:
            print("json.loads failed", output_text)
            # st.markdown("Error in extracting structured data", e)
            # st.markdown(output_text)
            gpt_extracted_infos = default_response

    return gpt_extracted_infos


def _postprocess_output(
    output_text: str,
    default_response: str,
    response_type: Literal["structured", "unstructured"],
):
    if response_type == "structured":
        gpt_extracted_infos = postprocess_structured_output(
            output_text, default_response
        )
    else:
        gpt_extracted_infos = output_text

    return gpt_extracted_infos

def _encode_image(image_path: str) -> str:
    """Encode an image to a base64 string if it is a valid image file."""
    if not os.path.exists(image_path):
        return image_path
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

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
    return_citations: bool = False,
    temperature: float = 0.0,
):

    final_model = general_pipelines[api_pipeline]["model"] if model is None else model
    url = general_pipelines[api_pipeline]["url"]

    payload = {
        "model": final_model,
        "messages": message,
        "temperature": temperature,
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
            if api_pipeline == "Perplexity":
                citations = clean_response["citations"]
            else:
                citations = None
            # st.markdown(output_text)
            # print("Worked")
        except Exception as e:
            print("GPT running failed", e, clean_response)
            output_text = str(default_response)

    gpt_extracted_infos = _postprocess_output(
        output_text, default_response, response_type
    )
    if return_citations and citations is not None:
        gpt_extracted_infos = {"citations": citations, "answer": gpt_extracted_infos}

    return gpt_extracted_infos


# Call chatGPT for all the given prompts in parallel.
async def _call_chatgpt_bulk(
    messages: List[List[Dict[str, str]]],
    default_response: str,
    response_type: Literal["structured", "unstructured"],
    model: str,
    api_key: str,
    api_pipeline: Literal["OpenAI", "Perplexity"],
    progress_bar: bool = True,
    additional_progress_bar_description: str = "",
    return_citations: bool = False,
    temperature: float = 0.0,
):
    final_progress_bar_description = "Processing Data"
    if additional_progress_bar_description != "":
        final_progress_bar_description += f" for {additional_progress_bar_description}"
    rate_limit = general_pipelines[api_pipeline]["rate_limit"]
    assert response_type in ["structured", "unstructured"]
    # print("Messages", messages)
    semaphore = asyncio.Semaphore(rate_limit)  # Control concurrency level
    async with aiohttp.ClientSession() as session:
        # Create a tqdm async progress bar
        if progress_bar:
            progress_bar = tqdm(
                total=len(messages), desc=final_progress_bar_description, position=0
            )

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
                    return_citations=return_citations,
                    temperature=temperature,
                )
            except Exception as e:
                print(f"Error in wrapped_call: {e}")
                return default_response
            finally:
                if progress_bar:
                    progress_bar.update(
                        1
                    )  # Update the progress for each completed task

        # Use asyncio.TaskGroup for managing tasks
        async with asyncio.TaskGroup() as tg:
            tasks = [
                tg.create_task(wrapped_call(session, message)) for message in messages
            ]
            responses = await asyncio.gather(*tasks)

        # Ensure the progress bar closes properly
        if progress_bar:
            progress_bar.close()

    return responses

def _get_model_id(
    system_prompt: str,
    model_name: str,
):
    return f"{model_name}-{system_prompt}"

def _create_ollama_models(
    system_prompts: List[str],
    model_name: str,
):
    saved_system_prompts_folder = os.path.join("data", "saved_system_prompts")
    os.makedirs(saved_system_prompts_folder, exist_ok=True)
    for i, system_prompt in enumerate(system_prompts):
        # print(system_prompt)
        model_id = f"{model_name}-{i}"
        ollama.create(
            model=model_id,
            system=system_prompt,
            from_=model_name,
            parameters={"temperature": 0.0},
        )


def _get_model_id(
    prompt: List[Dict[str, str]], system_prompts: List[str], model_name: str
):
    system_prompt = prompt[0]["content"]
    # compute the levenstein distance between the system prompt and the system prompts in the list
    prompt_id = min(
        range(len(system_prompts)),
        key=lambda x: levenshtein(system_prompts[x], system_prompt),
    )
    model_id = f"{model_name}-{prompt_id}"
    return model_id


def _ollama_inference(
    prompts: List[List[Dict[str, str]]],
    model_name: str,
    default_response: str,
    response_type: Literal["structured", "unstructured"],
    progress_bar: bool = True,
    additional_progress_bar_description: str = "",
    # stream: bool = False,
):
    final_progress_bar_description = "Processing Data"
    if additional_progress_bar_description != "":
        final_progress_bar_description += f" for {additional_progress_bar_description}"

    system_prompts = list(set([prompt[0]["content"] for prompt in prompts]))
    _create_ollama_models(system_prompts, model_name)

    answers = []
    if progress_bar:
        progress_bar = tqdm(
            total=len(prompts), desc=final_progress_bar_description, position=0
        )
    for prompt in prompts:
        model_id = _get_model_id(prompt, system_prompts, model_name)
        # if stream:
        #     stream = ollama.chat(model=model_id, messages=prompt[1:], stream=True)
        #     answer = ""
        #     for chunk in stream:
        #         new_chunk = chunk.message.content
        #         answer += new_chunk
        #         yield new_chunk
        #         # print(chunk.message.content, end="", flush=True)
        # else:
        answer = _do_ollama_inference(prompt, model_id, stream=False)
        answer = answer.message.content
        answer = _postprocess_output(answer, default_response, response_type)
        answers.append(answer)
        if progress_bar:
            progress_bar.update(1)

    return answers

def _do_ollama_inference(
    prompt: List[Dict[str, str]],
    model_id: str,
    stream: bool = False,
):
    final_user_prompt = prompt[1:]
    if "images" in final_user_prompt:
        final_images = []
        for image in final_user_prompt["images"]:
            final_images.append(f"data:image/png;base64,{_encode_image(image)}")
        final_user_prompt["images"] = final_images
        
    stream = ollama.chat(model=model_id, messages=final_user_prompt, stream=stream)
    return stream
        

def _ollama_inference_stream(
    prompts: List[List[Dict[str, str]]],
    model_name: str,
    default_response: str,
    response_type: Literal["structured", "unstructured"],
    custom_filter_function: Callable = None,
):
    answers = []

    system_prompts = list({prompt[0]["content"] for prompt in prompts})
    _create_ollama_models(system_prompts, model_name)

    def generator():

        for prompt in prompts:
            model_id = _get_model_id(prompt, system_prompts, model_name)
            stream = _do_ollama_inference(prompt, model_id, stream=True)
            answer = ""
            for chunk in stream:
                new_chunk = chunk.message.content
                new_chunk = replace_unneeded_characters(new_chunk)

                if custom_filter_function is not None:
                    new_chunk_is_relevant = custom_filter_function(answer)
                else:
                    new_chunk_is_relevant = True
                    
                if new_chunk_is_relevant:
                    yield new_chunk  # streaming each chunk
                    
                answer += new_chunk
            processed_answer = _postprocess_output(
                answer, default_response, response_type
            )
            answers.append(processed_answer)

    return generator(), answers


def get_answers(
    prompts: List[List[Dict[str, str]]],
    default_response: str,
    response_type: Literal["structured", "unstructured"],
    api_pipeline: Literal["OpenAI", "Perplexity", "Ollama"],
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    show_progress_bar: bool = True,
    additional_progress_bar_description: str = "",
    stream: bool = False,
    return_citations: bool = False,
    temperature: float = 0.0,
) -> List[Union[str, List[str], Dict[str, Union[str, float]]]]:

    assert (
        api_pipeline in general_pipelines.keys()
    ), "Invalid API pipeline, choose between OpenAI, Perplexity or Ollama."
    assert response_type in [
        "structured",
        "unstructured",
    ], "Invalid response type, choose between structured or unstructured."

    # try:
    if api_pipeline != "Ollama":
        # streaming is not applied for API pipelines other than Ollama
        assert api_key is not None, "API key is required for OpenAI or Perplexity"
        answers: Union[str, Dict[str, Union[str, float]]] = asyncio.run(
            _call_chatgpt_bulk(
                messages=prompts,
                response_type=response_type,
                model=model,
                default_response=default_response,
                api_key=api_key,
                api_pipeline=api_pipeline,
                progress_bar=show_progress_bar,
                additional_progress_bar_description=additional_progress_bar_description,
                return_citations=return_citations,
                temperature=temperature
            )  # _call_chatgpt_bulk(prompts, "{}", "structured")
        )

    else:
        if model is None:
            model = general_pipelines["Ollama"]["model"]

        os.system(f"ollama pull {model}")
        # Ollama call implementation is for now sequential to avoid memory issues
        answers = _ollama_inference(
            prompts=prompts,
            model_name=model,
            default_response=default_response,
            response_type=response_type,
            progress_bar=show_progress_bar,
            # stream=stream,
        )

    return answers


def get_answers_stream(
    prompts: List[List[Dict[str, str]]],
    default_response: str,
    response_type: Literal["structured", "unstructured"],
    api_pipeline: Literal["OpenAI", "Perplexity", "Ollama"],
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    custom_filter_function: Callable = None,
) -> List[Union[str, List[str], Dict[str, Union[str, float]]]]:

    if model is None:
        # only ollama is supported for streaming for now
        model = general_pipelines["Ollama"]["model"]

    os.system(f"ollama pull {model}")
    # Ollama call implementation is for now sequential to avoid memory issues
    generator, answers = _ollama_inference_stream(
        prompts=prompts,
        model_name=model,
        default_response=default_response,
        response_type=response_type,
        # stream=stream,
        custom_filter_function=custom_filter_function,
    )

    return generator, answers
