from inference import get_answers, get_answers_stream
from PIL import Image
import base64
import os


prompts = [
    [
        {
            "role": "system",
            "content": "Answer in JSON format with the following keys: 'answer', 'relevancy'.",
        },
        {"role": "user", "content": "What is the capital of France?"},
    ],
    [
        {
            "role": "system",
            "content": "Answer in JSON format with the following keys: 'answer', 'relevancy'.",
        },
        {"role": "user", "content": "What is the capital of Germany?"},
    ],
]

metadata_extraction_prompt = """This is a page of a document. I want to extract the document metadata from this page.
Extract the document publishing date, author organisations and the document title. 
Return only the results in a dictionnary JSON response without unnecessary spaces in the following format:
{
    "date": dd/mm/yyyy,
    "author": List[str]: The author organisations,
    "title": str: The title of the document
} 
If you cannot find any of the information, leave the field empty ('-').
Extract the information yourself and do not rely on any external library."""

def encode_image(image_path: str) -> str:
    """Encode an image to a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
one_page_path = "image.png"

image_prompts = [
    [
        {
            "role": "system",
            "content": metadata_extraction_prompt,
        },
        {
            "role": "user",
            "images": [one_page_path],
        },
    ]
]
# print(image_prompts)
# def custom_postprocess_function(stream_answer):
#     """
#     Determine if the streaming chunk is relevant for display.

#     This function analyzes partial JSON responses during streaming to determine
#     if the chunk contains content from the 'answer' field that should be displayed.

#     Args:
#         stream_answer (str): A chunk of the streaming response

#     Returns:
#         bool: True if the chunk is relevant for streaming display, False otherwise
#     """
#     # print("stream_answer", stream_answer)
#     # Check if we're in the answer field
#     if '"answer":' in stream_answer:

#         # If we're in the middle of the answer field but before relevancy
#         if '",' in stream_answer.replace(', "', ',"').replace('" ,', '",'):
#             # We're still in the answer field, so it's relevant
#             return False
#         else:
#             # print(stream_answer)
#             return True

#     else:
#         # If we can't clearly identify that we're in the answer field,
#         # consider it not relevant for streaming display
#         return False


if __name__ == "__main__":
    # answers = get_answers(
    #     prompts=image_prompts,
    #     default_response="{}",
    #     response_type="structured",
    #     api_pipeline="OpenAI",
    #     model="gpt-4o-mini",
    #     api_key=os.getenv("openai_api_key"),
    #     show_progress_bar=True,
    #     additional_progress_bar_description="test",
    #     # stream=stream,
    # )

    # print(answers)

    answers_stream, answers = get_answers_stream(
        prompts=image_prompts,
        default_response="{}",
        response_type="structured",
        api_pipeline="Ollama",
        model="gemma3:4b",
    )

    var_answer = ""

    for answer in answers_stream:
        var_answer += answer
        print(answer, end="", flush=True)

    print(answers)
