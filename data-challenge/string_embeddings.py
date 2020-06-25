import numpy as np
from scipy.sparse import csr_matrix, dok_matrix


def base2int(c):
    return {'A': 0, 'C': 1, 'G': 2, 'T': 3}.get(c, 0)

def kmer_to_spectral_index_multiplier(kmer_size, alphabet_size=4):
    '''
    Returns multiplier for indexing a k-mer
    
    Assuming kmer is a size k list of integers in [0, a-1],
        index(kmer) = sum(a**i * kmer_i)
    multiplier.dot(kmer) will give the index of kmer
    '''
    return alphabet_size**np.arange(kmer_size)

def kmer_to_mismatch_index_multiplier(kmer_size):
    '''
    Returns multiplier for the mismatch feature representation of a k-mer
    
    Mismatch feature representation of k-mer x:
        k (k-1)-mers xi, each corresponding to x with the ith element blanked
                     xi = x[0:i]x[i+1:k]
            
    base4_mismatch.dot(kmer) will give the indices of the mismatches
    '''
    base4_mismatch = np.zeros(kmer_size)
    base4_mismatch[1:] = kmer_to_spectral_index_multiplier(kmer_size)[:-1]
    base4_mismatch = base4_mismatch[None, :] * np.ones((kmer_size, kmer_size))
    for i in range(1, kmer_size):
        base4_mismatch[i] = base4_mismatch[i-1]
        base4_mismatch[i, i] = 0
        base4_mismatch[i, i-1] = base4_mismatch[i-1, i]
    return base4_mismatch

def kmer_decomposition(sample_strings, kmer_size=8, reverse=False):
    '''
    Returns the spectral embedding X of all strings in `sample_strings`
    
    X[index_string, index_k_mer] contains the number of occurrences
    in the string index_string of the k-mer index_k_mer 
    
    Input
    -----
    sample_strings: array-like of strings
    
    Output
    ------ 
    X is a sparse matrix of shape (n_samples, 4**k)
    '''
    n_rows = len(sample_strings)
    n_features = 4**kmer_size
    if reverse:
        n_features /= 2
        if kmer_size % 2 == 0:
            n_features += 2**kmer_size
    n_features = int(n_features)
    spectral_embedding = dok_matrix((n_rows, n_features))
    base4_multiplier = kmer_to_spectral_index_multiplier(kmer_size)
    
    for sample_index, sample_string in enumerate(sample_strings):
        input_bases = np.array([base2int(c) for c in sample_string])
        string_length = len(sample_string)
        for i in range(string_length-kmer_size+1):
            kmer = input_bases[i:i+kmer_size]
            kmer_index = base4_multiplier.dot(kmer)
            if reverse:
                index_reverse = base4_multiplier[::1].dot(3 - kmer)
                kmer_index = min(index_reverse, kmer_index)
            spectral_embedding[sample_index, kmer_index] += 1
    return csr_matrix(spectral_embedding)

def kmer_decomposition_mismatch(sample_strings, kmer_size=8):
    '''
    Returns the mismatch feature representation X of all strings
    in `sample_strings`
    
    X[index_string, index_blank * index_k_mer] contains the number of occurrences
     in the string index_string of the (k-1)-mer index_k_mer, with a blank at
     position index_blank
    
    Input
    -----
    
    Output
    ------   
    X is a sparse matrix of shape (n_samples, k*4**(k-1))  
    
    '''
    n_rows = len(sample_strings)
    stride = 4**(kmer_size-1)
    base4_mismatch_multiplier = kmer_to_mismatch_index_multiplier(kmer_size)

    mismatch_embedding = dok_matrix((n_rows, kmer_size * stride))
  
    for sample_index, sample_string in enumerate(sample_strings):
        input_bases = np.array([base2int(c) for c in sample_string])
        string_length = len(sample_string)
        # Loop over all k-mers
        for i in range(string_length-kmer_size+1):
            # Indices of all (k-1)-mers
            kmer = input_bases[i:i+kmer_size]
            kmer_indices = base4_mismatch_multiplier.dot(kmer)
            for block, kmer_index in enumerate(kmer_indices):
                kmer_index += block * stride 
                mismatch_embedding[sample_index, kmer_index] += 1
    return csr_matrix(mismatch_embedding)

def kmer_decomposition_positional(sample_strings, kmer_size=8):
    '''
    Supposes sequences are of the same length
    
    !Unfinished, untested!
    '''
    seq_length = len(sample_strings[0])
    n_positions = seq_length - kmer_size + 1
    print(n_positions)
    positional_embedding = []
    for i in range(n_positions):
        substrings = [s[i:i+kmer_size] for s in sample_strings]
        positional_embedding.append(kmer_decomposition(substrings))
    return positional_embedding