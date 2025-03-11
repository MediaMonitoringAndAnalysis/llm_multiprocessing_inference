from inference import get_answers

prompts = [
    [
        {"role": "system", "content": "Answer in JSON format with the following keys: 'capital', 'country'."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
    [
        {"role": "system", "content": "Answer in JSON format with the following keys: 'capital', 'country'."},
        {"role": "user", "content": "What is the capital of Germany?"},
    ],
]

stream = False

if __name__ == "__main__":
    answers = get_answers(
        prompts=prompts,
        default_response="{}",
        response_type="structured",
        api_pipeline="Ollama",
        model="phi4-mini:3.8b-q4_K_M",
        show_progress_bar=True,
        additional_progress_bar_description="test",
        # stream=stream,
    )
    
    # if stream:
        # for answer in answers:
        #     print(answer, end="", flush=True)
    # else:
        
    print(answers)
