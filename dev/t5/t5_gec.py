import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from load_t5 import tok, model, device

# Prefix tells T5 the task to conduct - grammar: implies grammatical error correction 
def top_k(sentence, k=5, prefix="grammar: "):
    # tokenise
    enc = tok(prefix + sentence, return_tensors="pt").to(device)


    with torch.no_grad():
        # Run inference with beam search to generate the top-k corrections (weights are not updated)
        out = model.generate(
            **enc,
            # specify k candidate outputs
            # A beam is a candidate partial sentence that T5 keeps while creating the final output
            # T5 fully extends them, scores them and keeps a ranking of them
            num_beams=k,
            num_return_sequences=k,
            max_new_tokens=128,
            return_dict_in_generate=True,
            output_scores=True,
        )

    # Convert the generated token IDs back into text, dropping special tokens like <pad> and </s>
    texts = tok.batch_decode(out.sequences, skip_special_tokens=True)
    return list(zip(texts, out.sequences_scores.tolist()))

