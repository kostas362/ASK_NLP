import re
from transformers import T5ForConditionalGeneration, T5Tokenizer
import spacy

#προτάσεις με αυτόματο που διαμόρφωσα
def auto_reconstruct(text):
    #kανόνες αυτόματου για ανακατασκευή
    text = re.sub(r"hope you too, to enjoy", "I hope you enjoy", text)
    text = re.sub(r"Thank your message", "Thank you for your message", text)
    text = re.sub(r"as his next contract checking", "regarding his upcoming contract review", text)
    text = re.sub(r"although bit delay", "although there was a slight delay", text)
    text = re.sub(r"to show our words to the doctor", "to forward our message to the doctor", text)
    text = re.sub(r"for paper and cooperation", "for the paper and collaboration", text)

    return text

#μοντέλο αγγλικών
nlp = spacy.load("en_core_web_sm")
#ανακατασκευη με spacy
def spacy_reconstruct(text):
    doc = nlp(text)
    reconstructed_sentences = []

    for sent in doc.sents:
        tokens = [token.text for token in sent if not token.is_punct or token.text == '.']
        sentence = ' '.join(tokens).strip()
        reconstructed_sentences.append(sentence)

    return ' '.join(reconstructed_sentences)

def spacy_auto_reconstruct(text):
    #καλει το auto_reconstruct και μετα θα καλέσει και το spacy
    text = auto_reconstruct(text)
    
    #κανει load το μοντέλο spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    reconstructed_sentences = []

    for sent in doc.sents:
        tokens = [token.text for token in sent if not token.is_punct or token.text == '.']
        sentence = ' '.join(tokens).strip()
        reconstructed_sentences.append(sentence)

    return ' '.join(reconstructed_sentences)

tokenizer = T5Tokenizer.from_pretrained("ramsrigouthamg/t5_paraphraser")
model = T5ForConditionalGeneration.from_pretrained("ramsrigouthamg/t5_paraphraser")

#ανακατασκευη με transformers
def transformers_reconstruct(text):
    input_text = "paraphrase: " + text.strip()
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

    outputs = model.generate(
        input_ids,
        max_length=128,
        num_beams=5,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        early_stopping=True
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    if not decoded or decoded.lower().startswith("paraphrase") or decoded == text.strip():
        return text.strip()
    return decoded

#σπαει το κειμενο σε chunks απ οτι θυμαμαι
def chunk_and_paraphrase(text, max_chunk_words=45):
    words = text.split()
    chunks = [' '.join(words[i:i+max_chunk_words]) for i in range(0, len(words), max_chunk_words)]
    full_output = []

    for idx, chunk in enumerate(chunks):
        print(f"\n🔹 Paraphrasing chunk {idx+1}:")
        result = transformers_reconstruct(chunk)
        print(f"[{idx+1}] {result}")
        full_output.append(result)

    return ' '.join(full_output)