from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

#Υπολογίζει την ομοιότητα συνημίτονου μεταξύ δύο κειμένων
def cosine_sim(text1, text2):
    vectorizer = CountVectorizer().fit_transform([text1, text2])
    return cosine_similarity(vectorizer[0], vectorizer[1])[0][0]

#Υπολογίζει την ομοιότητα συνημίτονου μεταξύ δύο ενσωματώσεων λέξεων
def cosine_sim_embeddings(embedding1, embedding2):
    return cosine_similarity(embedding1, embedding2)[0][0]

#φορτωνει το μοντέλο bert και τον tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

#για να παρουμε τις ενσοματλωσεις λέξεων απο το bert
def get_bert_embeddings(text):
    #κωδικοποίηση του κειμένου
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    #παίρνω τις ενσωματώσεις από το τελευταίο layer του BERT
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

#οπτικοποίηση των ενσωματώσεων λέξεων χρησιμοποιώντας PCA/t-SNE
def visualize_embeddings(embeddings):
    #PCA για μείωση διάστασης
    pca = PCA(n_components=8)
    pca_result = pca.fit_transform(embeddings)

    #t-SNE για μείωση διάστασης για οπτικοποίηση
    tsne = TSNE(n_components=2, random_state=42, perplexity=2)
    tsne_result = tsne.fit_transform(pca_result)

    #δημιουργία γραφήματος
    plt.figure(figsize=(12, 10))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=['blue', 'blue', 'red', 'red', 'green', 'green', 'orange', 'orange'],
                label=['Auto Text1', 'Auto Text2', 'T5 Text1', 'T5 Text2', 'spaCy Text1', 'spaCy Text2', 'Auto Text1 (spaCy)', 'Auto Text2 (spaCy)'])

    #το κέθε σημείο έχει μια ετικέτα
    for i, txt in enumerate(['Reconstructed Text1 (Auto)', 'Reconstructed Text2 (Auto)', 
                              'Reconstructed Text1 (Transformers)', 'Reconstructed Text2 (Transformers)', 
                              'Reconstructed Text1 (spaCy)', 'Reconstructed Text2 (spaCy)',
                              'Reconstructed Text1 (Auto + spaCy)', 'Reconstructed Text2 (Auto + spaCy)']):
        plt.annotate(txt, (tsne_result[i, 0], tsne_result[i, 1]))

    plt.title('t-SNE Ενσωματώσεων Λέξεων πριν και μετά την Ανακατασκευή')
    plt.legend(loc='upper left', fontsize=10)
    plt.show()