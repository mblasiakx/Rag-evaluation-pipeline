import numpy as np                                                                                                                                                                        
from sklearn.metrics.pairwise import cosine_similarity                                                                                                                                    
                                                                                                                                                                                            
def compute_anisotropy(embeddings: np.ndarray) -> float:                                                                                                                                  
      sim_matrix = cosine_similarity(embeddings)                                                                                                                                            
      upper = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]                                                                                                                             
      return float(np.mean(upper))