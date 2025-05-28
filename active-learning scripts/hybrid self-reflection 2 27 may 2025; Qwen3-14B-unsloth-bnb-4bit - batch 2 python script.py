def iterative_reflection(model, tokenizer, initial_prompt, iterations=66):
    conversation_history = []

    for i in range(iterations):
        print(f"\n{'='*50}")
        print(f"REFLECTION ITERATION {i+1}")
        print(f"{'='*50}")

        if i == 0:
            # First iteration: original prompt + reflection instruction
            messages = [
                {"role": "user", "content": f"{initial_prompt}\n\nPlease reflect deeply on this question. Think through multiple angles and perspectives."}
            ]
        else:
            # Subsequent iterations: build on previous reflections
            messages = conversation_history.copy()
            messages.append({
                "role": "user",
                "content": f"Based on your previous reflection, explore a different dimension or deeper aspect of the original question. What new insights emerge?"
            })

        # Generate response
        text = tokenizer.apply_chat_template(
            messages,
            tokenize = False,
            add_generation_prompt = True,
            enable_thinking = False,
        )

        # Capture output instead of streaming for conversation history
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors="pt").to("cuda")
            outputs = model.generate(
                **inputs,
                max_new_tokens = 32768,
                temperature = 0.7,
                top_p = 0.9,
                top_k = 40,
                do_sample = True,
                pad_token_id = tokenizer.eos_token_id
            )

        # Decode the response
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        print(response)

        # Add to conversation history
        messages.append({"role": "assistant", "content": response})
        conversation_history = messages

    # Final synthesis
    print(f"\n{'='*50}")
    print("SYNTHESIS & UNDERSTANDING")
    print(f"{'='*50}")

    final_messages = conversation_history.copy()
    final_messages.append({
        "role": "user",
        "content": "Now synthesize all your reflections. What is your final understanding of the original question? Provide a clear, concise summary of the key insights you've discovered through this reflection process."
    })

    final_text = tokenizer.apply_chat_template(
        final_messages,
        tokenize = False,
        add_generation_prompt = True,
        enable_thinking = False,
    )

    streamer = TextStreamer(tokenizer, skip_prompt=True)
    with torch.no_grad():
        _ = model.generate(
            **tokenizer(final_text, return_tensors="pt").to("cuda"),
            max_new_tokens = 32768,
            temperature = 0.6,
            top_p = 0.85,
            top_k = 30,
            streamer = streamer,
            pad_token_id = tokenizer.eos_token_id
        )

# Run the iterative reflection
initial_question = "meta-framing-mode:on. answer each query with few tokens. How can ethically aligned human-AI symbiosis unlock higher levels of individual and collective potential?"
iterative_reflection(model, tokenizer, initial_question, iterations=66)