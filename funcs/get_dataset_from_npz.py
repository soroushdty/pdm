import numpy as np
import json
from pathlib import Path

def get_dataset_from_npz(x_npz, y_train, y_test, merge_map_train_path, merge_map_test_path):
    """
    Maps embeddings from an NPZ object to train/test arrays using merge maps.
    
    Args:
        x_npz: The loaded NPZ object containing {item_name: embedding_vector}.
        y_train: Numpy array for training labels.
        y_test: Numpy array for testing labels.
        merge_map_train_path: Path to 'merge_map_train.json'.
        merge_map_test_path: Path to 'merge_map_test.json'.
        
    Returns:
        Tuple of (X_train, X_test, Y_train, Y_test) as numpy arrays.
    """
    
    # Load the merge maps which contain "Item" names for each merged row index
    with open(merge_map_train_path, 'r') as f:
        train_map = json.load(f)
    with open(merge_map_test_path, 'r') as f:
        test_map = json.load(f)


    def build_x_matrix(x_npz, y_data, df, item_col_name):
        num_rows = y_data.shape[0]
        # Get embedding dimension from the first embedding in x_npz
        embedding_dim = next(iter(x_npz.values())).shape[0]
        X_matrix = np.zeros((num_rows, embedding_dim))

        for i in range(num_rows):
            # The first column of y_data contains the original index from the preprocessed dataframe
            original_idx = int(y_data[i, 0])
        
            # Get the Item name from the preprocessed DataFrame using the original_idx
            # Use .loc to ensure index-based retrieval from the DataFrame
            item_name = df.loc[original_idx, item_col_name]
        
            # Retrieve the embedding from the NPZ object using the Item name
            embedding = x_npz[item_name]
            X_matrix[i] = embedding
    return X_matrix
    
    # Generate X matrices by aligning embeddings with the order of Y labels
    X_train = build_x_matrix(y_train, train_map)
    X_test = build_x_matrix(y_test, test_map)

    # Y_train and Y_test are returned as provided, now aligned with X_train and X_test
    return X_train, X_test, y_train, y_test
