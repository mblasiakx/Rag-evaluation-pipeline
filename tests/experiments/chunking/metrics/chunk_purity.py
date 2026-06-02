from sentence_transformers import util
from .model_registry import get_model

#sprawdza, czy chunk zawiera jeden temat, czy kilka. A nie jak coherence ze patrzy na relacje miedyz zdaniami
#jeśli wszystkie zdania należą do jednego klastra semantycznego → wysoka purity
def chunk_purity(chunks):
    purity_scores = []
    for chunk in chunks:
         sentences = [s.strip() for s in chunk.split(".") if len(s.strip()) > 0]
         if len(sentences)<2:
              purity_scores.append(1.0)
              continue
        
         embeddings = get_model().encode(sentences, convert_to_tensor=True)
         sim_matrix = util.cos_sim(embeddings, embeddings)

         n = sim_matrix.shape[0]
         mean_sim = (sim_matrix.sum() - n) / (n * (n - 1))

         purity_scores.append(mean_sim.item())

    
    return sum(purity_scores) / len(purity_scores)