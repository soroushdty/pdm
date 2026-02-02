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

    def build_x_matrix(y_data, merge_map):
        x_list = []
        # Each row in y represents a merged Patient-Item group
        for i in range(len(y_data)):
            # Get the Item name associated with this merged row index from the map
            # Note: merge_map keys are strings of the row index
            item_name = merge_map[str(i)]
            
            # Retrieve the embedding from the NPZ object using the Item name
            if item_name in x_npz:
                x_list.append(x_npz[item_name])
            else:
                # Fallback: handle cases where the item might be missing (should not happen with consistent data)
                embedding_dim = x_npz[list(x_npz.keys())[0]].shape[0]
                x_list.append(np.zeros(embedding_dim))
                
        return np.array(x_list)

    # Generate X matrices by aligning embeddings with the order of Y labels
    X_train = build_x_matrix(y_train, train_map)
    X_test = build_x_matrix(y_test, test_map)

    # Y_train and Y_test are returned as provided, now aligned with X_train and X_test
    return X_train, X_test, y_train, y_test
