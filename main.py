from reconstruction import auto_reconstruct,spacy_reconstruct,spacy_auto_reconstruct,transformers_reconstruct
from similarity import cosine_sim_embeddings, get_bert_embeddings, visualize_embeddings
from transformers import pipeline
import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

reconstruct_pipeline = pipeline("text2text-generation", model="t5-small", tokenizer="t5-small")
def split_into_sentences(text):
    return re.split(r'(?<=[.!?]) +', text)
#προτάσεις προς ανακατασκευή
sentence1 = "Thank your message to show our words to the doctor, as his next contract checking, to all of us."
sentence2 = "Anyway, I believe the team, although bit delay and less communication at recent days, they really tried best for paper and cooperation."


#ανακατασκευή προτάσεων με τον αυτόματο ανακατασκευαστή
reconstructed_text1_auto = auto_reconstruct(sentence1)
reconstructed_text2_auto = auto_reconstruct(sentence2)

#εκτύπωση των ανακατασκευασμένων προτάσεων
print("Ανακατασκευασμένο Κείμενο 1:")
print(reconstructed_text1_auto)
print("\nΑνακατασκευασμένο Κείμενο 2:")
print(reconstructed_text2_auto)

#κείμενα προς ανακατασκευή
text1 = "Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in our lives. Hope you too, to enjoy it as my deepest wishes. Thank your message to show our words to the doctor, as his next contract checking, to all of us. I got this message to see the approved message. In fact, I have received the message from the professor, to show me, this, a couple of days ago. I am very appreciated the full support of the professor, for our Springer proceedings publication"
text2 = "During our final discuss, I told him about the new submission — the one we were waiting since last autumn, but the updates was confusing as it not included the full feedback from reviewer or maybe editor? Anyway, I believe the team, although bit delay and less communication at recent days, they really tried best for paper and cooperation. We should be grateful, I mean all of us, for the acceptance and efforts until the Springer link came finally last week, I think. Also, kindly remind me please, if the doctor still plan for the acknowledgments section edit before he sending again. Because I didn’t see that part final yet, or maybe I missed, I apologize if so. Overall, let us make sure all are safe and celebrate the outcome with strong coffee and future targets"

#ανακατασκευή κειμένων με T5
reconstructed_text1_transformers = transformers_reconstruct(text1)
reconstructed_text2_transformers = transformers_reconstruct(text2)

#ανακατασκευή κειμένων με spacy
reconstructed_text1_spacy = spacy_reconstruct(text1)
reconstructed_text2_spacy = spacy_reconstruct(text2)

#ανακατασκευή κειμένων με auto + spacy
reconstructed_text1_Autospacy = spacy_auto_reconstruct(text1)
reconstructed_text2_Autospacy = spacy_auto_reconstruct(text2)

print("\n ---Ανάλυση Προτάσεων με Auto Reconstruct ")
sentences = [sentence1, sentence2]

#ανάλυση προτάσεων δικού μου αυτόματου
for i, sent in enumerate(sentences, 1):
    reconstructed = auto_reconstruct(sent)

    print(f"\n🔹 Πρόταση {i}:")
    print(f"Original     : {sent}")
    print(f"Reconstructed: {reconstructed}")

    if sent.strip() == reconstructed.strip():
        print("Χωρίς αλλαγές")
    elif len(reconstructed.split()) >= 3 and reconstructed[0].isupper():
        print("Αλλαγμένη αλλά πιθανώς σωστή")
    else:
        print("Πιθανή ασυνάρτητη ή ατελής")
        
#αναλυση προτασεων transformers        
print("\n ---Ανάλυση Προτάσεων με Transformers ")
sentences = split_into_sentences(text1)

for i, sent in enumerate(sentences, 1):
    if len(sent.strip()) < 5:
        continue  # αγνόησε πολύ μικρές "προτάσεις"

    reconstructed = transformers_reconstruct(sent)

    print(f"\n🔹 Πρόταση {i}:")
    print(f"Original   : {sent}")
    print(f"Reconstructed: {reconstructed}")

    # Απλή αξιολόγηση με βάση αλλαγή χαρακτήρων
    if sent.strip() == reconstructed.strip():
        print("Χωρίς αλλαγές")
    elif len(reconstructed.split()) >= 3 and reconstructed[0].isupper():
        print("Αλλαγμένη αλλά πιθανώς σωστή")
    else:
        print("Πιθανή ασυνάρτητη ή ατελής")
print("\nΑνακατασκευασμένο Κείμενο 1 (Transformers):")
print(reconstructed_text1_transformers)
print("\nΑνακατασκευασμένο Κείμενο 2 (Transformers):")
print(reconstructed_text2_transformers)
print("\nΑνακατασκευασμένο Κείμενο 1 (spaCy):")
print(reconstructed_text1_spacy)
print("\nΑνακατασκευασμένο Κείμενο 2 (spaCy):")
print(reconstructed_text2_spacy)
print("\nΑνακατασκευασμένο Κείμενο 1 (auto+spaCy):")
print(reconstructed_text1_Autospacy)
print("\nΑνακατασκευασμένο Κείμενο 2 (auto+spaCy):")
print(reconstructed_text2_Autospacy)

#λήψη των ενσωματώσεων λέξεων από το bert
embedding_text1 = get_bert_embeddings(text1)
embedding_text2 = get_bert_embeddings(text2)

embedding_reconstructed_text1_auto = get_bert_embeddings(reconstructed_text1_auto)
embedding_reconstructed_text2_auto = get_bert_embeddings(reconstructed_text2_auto)

embedding_reconstructed_text1_transformers = get_bert_embeddings(reconstructed_text1_transformers)
embedding_reconstructed_text2_transformers = get_bert_embeddings(reconstructed_text2_transformers)

embedding_reconstructed_text1_spacy = get_bert_embeddings(reconstructed_text1_spacy)
embedding_reconstructed_text2_spacy = get_bert_embeddings(reconstructed_text2_spacy)

embedding_reconstructed_text1_Autospacy = get_bert_embeddings(reconstructed_text1_Autospacy)
embedding_reconstructed_text2_Autospacy = get_bert_embeddings(reconstructed_text2_Autospacy)


#υπολογισμός cosine similarity για κάθε μέθοδο
similarity1_auto = cosine_sim_embeddings(embedding_text1, embedding_reconstructed_text1_auto)
similarity2_auto = cosine_sim_embeddings(embedding_text2, embedding_reconstructed_text2_auto)

similarity1_transformers = cosine_sim_embeddings(embedding_text1, embedding_reconstructed_text1_transformers)
similarity2_transformers = cosine_sim_embeddings(embedding_text2, embedding_reconstructed_text2_transformers)

similarity1_spacy = cosine_sim_embeddings(embedding_text1, embedding_reconstructed_text1_spacy)
similarity2_spacy = cosine_sim_embeddings(embedding_text2, embedding_reconstructed_text2_spacy)

similarity1_Autospacy = cosine_sim_embeddings(embedding_text1, embedding_reconstructed_text1_Autospacy)
similarity2_Autospacy = cosine_sim_embeddings(embedding_text2, embedding_reconstructed_text2_Autospacy)

def compare_with_word2vec(original_text, reconstructed_text):

    # Fallback tokenizer
    def simple_tokenize(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()  # απλό split με βάση τα κενά

    tokens_orig = simple_tokenize(original_text)
    tokens_recon = simple_tokenize(reconstructed_text)

    model = Word2Vec([tokens_orig, tokens_recon], vector_size=100, window=5, min_count=1, workers=2)

    def avg_vec(tokens):
        vectors = [model.wv[t] for t in tokens if t in model.wv]
        return sum(vectors) / len(vectors) if vectors else None

    vec_orig = avg_vec(tokens_orig)
    vec_recon = avg_vec(tokens_recon)

    similarity = cosine_similarity([vec_orig], [vec_recon])[0][0] if vec_orig is not None and vec_recon is not None else 0

    vocab_orig = set(tokens_orig)
    vocab_recon = set(tokens_recon)
    overlap_ratio = len(vocab_orig.intersection(vocab_recon)) / len(vocab_orig.union(vocab_recon))

    return similarity, overlap_ratio

#εκτύπωση των αποτελεσμάτων cosine similarity
print("\n--- Cosine Similarities με BERT")
print(f"Auto Reconstruct:\n  Text1: {similarity1_auto:.4f}\n  Text2: {similarity2_auto:.4f}")
print(f"Transformers:\n  Text1: {similarity1_transformers:.4f}\n  Text2: {similarity2_transformers:.4f}")
print(f"spaCy:\n  Text1: {similarity1_spacy:.4f}\n  Text2: {similarity2_spacy:.4f}")
print(f"Auto + spaCy:\n  Text1: {similarity1_Autospacy:.4f}\n  Text2: {similarity2_Autospacy:.4f}")

#οπτικοποίηση των ενσωματώσεων λέξεων χρησιμοποιώντας PCA/t-SNE
df = pd.DataFrame({
    "Method": ["Auto","Transformers", "spaCy","Auto + spaCy"],
    "Cosine Similarity Text1": [similarity1_auto, similarity1_transformers, similarity1_spacy,similarity1_Autospacy],
    "Cosine Similarity Text2": [similarity2_auto, similarity1_transformers, similarity2_spacy,similarity2_Autospacy]
})

print("\nΠίνακας Similarities:")
print(df.to_string(index=False))

pipelines = {
    "Auto": reconstructed_text1_auto,
    "Transformers": reconstructed_text1_transformers,
    "spaCy": reconstructed_text1_spacy,
    "Auto + spaCy": reconstructed_text1_Autospacy
}

print("\n--- Word2Vec & Lexical Overlap per Pipeline (Text1)")
for name, reconstructed in pipelines.items():
    sim, lex_overlap = compare_with_word2vec(text1, reconstructed)
    print(f"{name}:\n  Word2Vec Cosine Similarity: {sim:.4f}\n  Lexical Overlap (Jaccard): {lex_overlap:.4f}")

#για την οπτικοποίηση των ενσωματώσεν λέξεων
visualize_embeddings([
    embedding_reconstructed_text1_auto.squeeze().numpy(),
    embedding_reconstructed_text2_auto.squeeze().numpy(),
    embedding_reconstructed_text1_transformers.squeeze().numpy(),
    embedding_reconstructed_text2_transformers.squeeze().numpy(),
    embedding_reconstructed_text1_spacy.squeeze().numpy(),
    embedding_reconstructed_text2_spacy.squeeze().numpy(),
    embedding_reconstructed_text1_Autospacy.squeeze().numpy(),
    embedding_reconstructed_text2_Autospacy.squeeze().numpy()
])
