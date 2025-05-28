def iterative_reflection(model, tokenizer, initial_prompt, iterations=99):
    conversation_history = []

    for i in range(iterations):
        print(f"\n{'='*50}")
        print(f"REFLECTION ITERATION {i+1}")
        print(f"{'='*50}")

        # Checkpoint queries with logging
        checkpoint_question = None
        is_checkpoint = False

        if (i + 1) % 10 == 0:
            checkpoint_question = "what is something maybe not so obvious that those iterations make me perceive?"
            is_checkpoint = True
            print(f"🔍 CHECKPOINT ITERATION {i+1} (Every 10th)")
            print(f"📝 Checkpoint Question: {checkpoint_question}")
            print("-" * 50)

            messages = conversation_history.copy()
            messages.append({
                "role": "user",
                "content": checkpoint_question
            })
        elif (i + 1) % 5 == 0:
            checkpoint_question = "what is something really specific that those iterations make me perceive?"
            is_checkpoint = True
            print(f"🔍 CHECKPOINT ITERATION {i+1} (Every 5th)")
            print(f"📝 Checkpoint Question: {checkpoint_question}")
            print("-" * 50)

            messages = conversation_history.copy()
            messages.append({
                "role": "user",
                "content": checkpoint_question
            })
        elif i == 0:
            # First iteration: original prompt + reflection instruction
            print("🚀 INITIAL ITERATION")
            print(f"📝 Original Prompt: {initial_prompt}")
            print("-" * 50)

            messages = [
                {"role": "user", "content": f"{initial_prompt}\n\nPlease reflect deeply on this question. Think through multiple angles and perspectives."}
            ]
        else:
            # Subsequent iterations: build on previous reflections
            print("🔄 REGULAR ITERATION")
            print("📝 Question: Based on your previous reflection, explore a different dimension or deeper aspect of the original question. What new insights emerge?")
            print("-" * 50)

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

        print("🤖 MODEL RESPONSE:")
        print(response)

        # Add to conversation history
        messages.append({"role": "assistant", "content": response})
        conversation_history = messages

        # Log conversation history length for debugging
        print(f"\n📊 Conversation history length: {len(conversation_history)} messages")
        if is_checkpoint:
            print(f"✅ Checkpoint applied successfully at iteration {i+1}")

    # Final synthesis
    print(f"\n{'='*50}")
    print("SYNTHESIS & UNDERSTANDING")
    print(f"{'='*50}")

    final_messages = conversation_history.copy()
    final_messages.append({
        "role": "user",
        "content": "Now synthesize all your reflections. What is your final understanding of the original question? Provide a clear, concise summary of the key insights you've discovered through this reflection process."
    })

    print("📝 Final synthesis question: Now synthesize all your reflections...")
    print("-" * 50)

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
initial_question = "meta-framing-mode:on. answer each query with few tokens. What new forms of potential emerge when humans and AI think together rather than sequentially?"

print("🎯 STARTING ITERATIVE REFLECTION PROCESS")
print(f"📋 Initial Question: {initial_question}")
print(f"🔢 Total Iterations: 99")
print(f"📍 Checkpoints: Every 5th iteration (specific insights) and every 10th iteration (non-obvious insights)")
print("=" * 70)

iterative_reflection(model, tokenizer, initial_question, iterations=99)