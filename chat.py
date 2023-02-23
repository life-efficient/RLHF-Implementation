def chat(tokenizer, model):
    """Chat with the model."""
    prompt = ""
    while True:
        # GET USER INPUT
        next_input = "You: " + input("You: ") + "\nBot: "
        print(next_input)
        prompt += next_input

        # GENERATE A SEQUENCE OF TOKENS USING THE MODEL'S FORWARD METHOD
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        output = model.generate(
            input_ids, max_length=100, do_sample=True, top_k=50, top_p=0.95, temperature=0.7)

        # PRINT THE RESPONSE AND UPDATE THE PROMPT
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        print(response)
        prompt += response
