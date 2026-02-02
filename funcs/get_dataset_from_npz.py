import numpy as np
import json

def get_dataset_from_npz(x_npz, y_train, y_test, merge_map_train_path, merge_map_test_path):
    """
    Maps embeddings from an NPZ object to train/test arrays using merge maps.
    """
    
    # Load the merge maps
    with open(merge_map_train_path, 'r') as f:
        train_map = json.load(f)
    with open(merge_map_test_path, 'r') as f:
        test_map = json.load(f)

    def build_x_matrix(y_data, merge_map):
        # Determine embedding dimension from the first available key
        sample_key = list(x_npz.keys())[0]
        emb_dim = x_npz[sample_key].shape[0]
        
        # Initialize the matrix with zeros
        X_matrix = np.zeros((len(y_data), emb_dim))
        
        for i in range(len(y_data)):
            # Map index to Item name via the JSON map
            item_name = merge_map.get(str(i))
            
            if item_name and item_name in x_npz:
                X_matrix[i] = x_npz[item_name]
            # If item is missing, X_matrix[i] remains zeros
                
        return X_matrix

    # Correctly initialize and assign the matrices
    X_train = build_x_matrix(y_train, train_map)
    X_test = build_x_matrix(y_test, test_map)

    return X_train, X_test, y_train, y_test
