import os
import logging

ROOT = os.path.abspath(os.path.dirname(__file__))

SYSTEM_INSTRUCT = """
        You are a helpful assistant that answers user questions.
        You are given a set of documents for reference.
        If you cannot answer on you own, use the documents as reference.
        Do not hallucinate, give concise answers and do not deviate from the question subject
        """


def get_docs_asqa(sample, n):
    """ Returns the top-n reference documents from the 'ctxs' key concatenated on 
    a string. 

    n: <int>
    sample: <dict>
    text: <str>
    """
    texts = [f"{doc['title']} {doc['text']}" for doc in sample["ctxs"][:n]]
    text = "\n".join(texts)
    return text

def setup_logger(name):
    logging.basicConfig(
        filename=f"{name}.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(f"{name}")


def summarize_docs(model, tokenizer, docs, device, doc_n=3):

    """ Receives a huggingface transformer model and docs to summarize
    Returns the summary of the docs
    """

    logger = setup_logger("summarizer")

    system_instruct= f"""
    You will receive {doc_n} documents and you need to summarize them mantaining the most relevant information.
    These documents may refer to multiple entities, if so, format the output so information always refers to the
    correct entity.
    Do not add information, do not hallucinate facts, do not add new entities."""


    instruct_dict = [{"role" : "system", "content": system_instruct},
                             {"role": "user", "content": docs}]

    cur_prompt = tokenizer.apply_chat_template( instruct_dict,
                                                tokenize=False,
                                                add_generation_prompt=True,
    )

    tokens = tokenizer(cur_prompt,
                            return_tensors="pt",
                            padding=True,
                            truncation=True)

    cur_iids = tokens.input_ids.to(device)
    attention_mask = tokens.attention_mask.to(device)

    out = model.generate(
        cur_iids,
        attention_mask=attention_mask,                  # Pass the attention mask
        pad_token_id=tokenizer.eos_token_id,            # Set pad_token_id to eos_token_id
        return_dict_in_generate=True,
        max_new_tokens=256,
    )

    gen_tokens = out["sequences"]
    sum_text = tokenizer.decode(
        gen_tokens[0, cur_iids.shape[-1]:],
        skip_special_tokens=True
    )
    logger.info("Generated Summary: %s", sum_text)
    return sum_text
