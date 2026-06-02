from tests.experiments.retrieval.metrics.hit import is_hit
from tests.experiments.retrieval.metrics.precision_at_k import compute_precision_at_k   
from tests.experiments.retrieval.metrics.hit_rate import compute_hit_rate
from tests.experiments.retrieval.metrics.mrr import compute_mrr
import numpy as np

#ishit

def test_is_hit_chunk_before_offset_returns_false():
      chunk = {"start_char": 0, "end_char": 10}
      query_offsets = [{"start_char": 20, "end_char": 50}]
      assert is_hit(query_offsets, chunk) == False

def test_is_hit_chunk_starts_before_offset_returns_true():
      chunk = {"start_char": 10, "end_char": 30}
      query_offsets = [{"start_char": 20, "end_char": 50}]
      assert is_hit(query_offsets, chunk) == True

def test_is_hit_chunk_starts_in_the_middle_of_offset_returns_true():
      chunk = {"start_char": 40, "end_char": 70}
      query_offsets = [{"start_char": 20, "end_char": 50}]
      assert is_hit(query_offsets, chunk) == True

def test_is_hit_chunk_contains_offset_returns_true():
      chunk = {"start_char": 10, "end_char": 60}
      query_offsets = [{"start_char": 20, "end_char": 50}]
      assert is_hit(query_offsets, chunk) == True

def test_is_hit_chunk_equal_to_offset_returns_true():
      chunk = {"start_char": 10, "end_char": 60}
      query_offsets = [{"start_char": 10, "end_char": 60}]
      assert is_hit(query_offsets, chunk) == True

def test_is_hit_chunk_after_offset_returns_false():
      chunk = {"start_char": 55, "end_char": 70}
      query_offsets = [{"start_char": 20, "end_char": 50}]
      assert is_hit(query_offsets, chunk) == False

def test_is_hit_chunk_end_equal_to_offset_start_returns_false():
      chunk = {"start_char": 10, "end_char": 30}
      query_offsets = [{"start_char": 30, "end_char": 50}]
      assert is_hit(query_offsets, chunk) == False



#compute_precision_at_k

                                                                                                                     
                                                                                                                                                                                                         
def test_precision_at_k_all_hits_returns_one():                                                                                                                                                              
      query_offsets = [{"start_char": 20, "end_char": 50}]                                                                                                                                                     
      retrieved_chunks = [                                                                                                                                                                                     
          {"start_char": 25, "end_char": 35},                                                                                                                                                                  
          {"start_char": 30, "end_char": 45},
      ]                                                                                                                                                                                                        
      assert compute_precision_at_k(retrieved_chunks, query_offsets, k=2) == 1.0 

def test_precision_at_k_no_hits_returns_zero():                                                                                                                                                              
      query_offsets = [{"start_char": 20, "end_char": 50}]                                                                                                                                                     
      retrieved_chunks = [                                                                                                                                                                                     
          {"start_char": 10, "end_char": 15},                                                                                                                                                                  
          {"start_char": 60, "end_char": 80},
      ]                                                                                                                                                                                                        
      assert compute_precision_at_k(retrieved_chunks, query_offsets, k=2) == 0.0 


def test_precision_at_k_one_of_two_hits_returns_half():                                                                                                                                                             
      query_offsets = [{"start_char": 20, "end_char": 50}]                                                                                                                                                     
      retrieved_chunks = [                                                                                                                                                                                     
          {"start_char": 30, "end_char": 40},                                                                                                                                                                  
          {"start_char": 60, "end_char": 80},
      ]                                                                                                                                                                                                        
      assert compute_precision_at_k(retrieved_chunks, query_offsets, k=2) == 0.5            


def test_precision_at_k_zero_k_returns_zero():                                                                                                                                                               
      query_offsets = [{"start_char": 20, "end_char": 50}]                                                                                                                                                     
      retrieved_chunks = [{"start_char": 25, "end_char": 35}]                                                                                                                                                  
      assert compute_precision_at_k(retrieved_chunks, query_offsets, k=0) == 0         


#compute_hit_rate

def test_compute_hit_rate_returns_one():                                                                                                                                                                     
      query_offsets = [{"start_char": 20, "end_char": 50}]                                                                                                                                                     
      retrieved_chunks = [                                                                                                                                                                                     
          {"start_char": 60, "end_char": 80},                                                                                                                                                                  
          {"start_char": 25, "end_char": 35},                                                                                                                                                                  
      ]
      assert compute_hit_rate(retrieved_chunks, query_offsets) == 1.0    
      

    
def test_compute_hit_rate_returns_zero():                                                                                                                                                                     
      query_offsets = [{"start_char": 20, "end_char": 50}]                                                                                                                                                     
      retrieved_chunks = [                                                                                                                                                                                     
          {"start_char": 60, "end_char": 80},                                                                                                                                                                  
          {"start_char": 5, "end_char": 15},                                                                                                                                                                  
      ]
      assert compute_hit_rate(retrieved_chunks, query_offsets) == 0.0    

def test_compute_hit_rate_empty_list():
      query_offsets = [{"start_char": 20, "end_char": 50}]                                                                                                                                                     
      retrieved_chunks = []
      assert compute_hit_rate(retrieved_chunks, query_offsets) == 0.0    

#compute_mr

def test_compute_mrr_first_chunk_is_hit_returns_one():                                                                                                                                                       
      query_offsets = [{"start_char": 20, "end_char": 50}]                                                                                                                                                     
      retrieved_chunks = [                                                                                                                                                                                     
          {"start_char": 25, "end_char": 45},                                                                                                                                                                  
          {"start_char": 60, "end_char": 80},
          {"start_char": 90, "end_char": 110},                                                                                                                                                                 
      ]           
      assert compute_mrr(retrieved_chunks, query_offsets) == 1.0

def test_compute_mrr_second_chunk_is_hit_returns_half():                                                                                                                                                      
      query_offsets = [{"start_char": 20, "end_char": 50}]                                                                                                                                                     
      retrieved_chunks = [                                                                                                                                                                                     
          {"start_char": 15, "end_char": 19},                                                                                                                                                                  
          {"start_char": 25, "end_char": 40},
          {"start_char": 90, "end_char": 110},                                                                                                                                                                 
      ]           
      assert compute_mrr(retrieved_chunks, query_offsets) == 0.5

def test_compute_mrr_last_chunk_is_hit_returns_third():                                                                                                                                                    
      query_offsets = [{"start_char": 20, "end_char": 50}]                                                                                                                                                     
      retrieved_chunks = [                                                                                                                                                                                     
          {"start_char": 15, "end_char": 19},                                                                                                                                                                  
          {"start_char": 60, "end_char": 80},
          {"start_char": 30, "end_char": 40},                                                                                                                                                                 
      ]           
      assert np.isclose(compute_mrr(retrieved_chunks, query_offsets), 1/3)   


def test_compute_mrr_no_chunk_is_hit_returns_zero():                                                                                                                                                    
      query_offsets = [{"start_char": 20, "end_char": 50}]                                                                                                                                                     
      retrieved_chunks = [                                                                                                                                                                                     
          {"start_char": 10, "end_char": 15},                                                                                                                                                                  
          {"start_char": 55, "end_char": 80},
          {"start_char": 60, "end_char": 80},                                                                                                                                                                 
      ]           
      assert compute_mrr(retrieved_chunks, query_offsets) == 0.0   
