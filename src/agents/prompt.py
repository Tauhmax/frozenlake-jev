from src.agents.actions import DIRECTIONS, LETTERS


def format_prompt(board, gamma):
    choices = "\n".join(
        f"{letter}: {name}" for letter, name in zip(LETTERS, DIRECTIONS)
    )
    return (
        "You are playing deterministic FrozenLake.\n"
        "P is you, F is safe floor, H is a terminal hole, G is the terminal goal.\n"
        "Moving beyond an edge leaves you in place. Reward is 1 on entering G, "
        f"otherwise 0. Maximize discounted return with discount {gamma}; "
        "prefer a shorter safe path to G.\n"
        "Choose exactly one letter A, B, C, or D. Do not explain.\n\n"
        f"Map:\n{board}\n\nActions:\n{choices}\n"
    )


def prepare_choice_tokens(tokenizer, prompt):
    if tokenizer.chat_template:
        prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if prefix.count("<think>") > prefix.count("</think>"):
            raise ValueError("Chat template leaves an open thinking block")
    else:
        prefix = prompt
    # A newline prevents the colon from merging with the next choice token.
    prefix += "Answer:\n"
    input_ids = tokenizer.encode(prefix, add_special_tokens=False)
    choices = []
    for letter in LETTERS:
        extended = tokenizer.encode(prefix + letter, add_special_tokens=False)
        if extended[:-1] != input_ids or len(extended) != len(input_ids) + 1:
            raise ValueError(
                f"{letter} is not a single next token at the answer position"
            )
        choices.append(extended[-1])
    if len(set(choices)) != 4:
        raise ValueError("Choice token IDs must be distinct")
    return prefix, input_ids, choices
