from inference import get_answers, get_answers_stream

prompts = [
    [
        {"role": "system", "content": "Answer in JSON format with the following keys: 'answer', 'relevancy'."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
    [
        {"role": "system", "content": "Answer in JSON format with the following keys: 'answer', 'relevancy'."},
        {"role": "user", "content": "What is the capital of Germany?"},
    ],
]

def custom_postprocess_function(stream_answer):
    """
    Determine if the streaming chunk is relevant for display.

    This function analyzes partial JSON responses during streaming to determine
    if the chunk contains content from the 'answer' field that should be displayed.

    Args:
        stream_answer (str): A chunk of the streaming response

    Returns:
        bool: True if the chunk is relevant for streaming display, False otherwise
    """
    # print("stream_answer", stream_answer)
    # Check if we're in the answer field
    if '"answer":' in stream_answer:

        # If we're in the middle of the answer field but before relevancy
        if '",' in stream_answer.replace(', "', ',"').replace('" ,', '",'):
            # We're still in the answer field, so it's relevant
            return False
        else:
            # print(stream_answer)
            return True
    
    else:
        # If we can't clearly identify that we're in the answer field,
        # consider it not relevant for streaming display
        return False


if __name__ == "__main__":
    # answers = get_answers(
    #     prompts=prompts,
    #     default_response="{}",
    #     response_type="structured",
    #     api_pipeline="Ollama",
    #     model="phi4-mini:3.8b-q4_K_M",
    #     show_progress_bar=True,
    #     additional_progress_bar_description="test",
    #     # stream=stream,
    # )
    
    # print(answers)
    
    answers_stream, answers = get_answers_stream(
        prompts=prompts,
        default_response="{}",
        response_type="structured",
        api_pipeline="Ollama",
        model="phi4-mini:3.8b-q4_K_M",
    )
    
    var_answer = ""
    
    for answer in answers_stream:
        var_answer += answer
        print(answer, end="", flush=True)
        
        
    print(answers)
