import sentencepiece as spm

sp = spm.SentencePieceProcessor(model_file='../data/processed/spm_en_id.model')

sentences = [
    "Hello World!",
    "How are you?"
] 

for sentence in sentences:
    print(f"Original: {sentence}")

    tokens = sp.encode(sentence, out_type=str)
    print(f"Tokens: {tokens}")

    ids = sp.encode(sentence, out_type=int)
    print(f"IDs: {ids}")

    output_sentence = sp.decode(ids)
    print(f"Decoded: {output_sentence}")
    print()