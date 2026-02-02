import json
import numpy as np

def get_dataset_from_npz(
    x_npz,
    y_train,
    y_test,
    merge_map_train_path,
    merge_map_test_path,
):
    """
    Build X_train and X_test feature matrices from an embeddings NPZ and merge maps.

    Index alignment
    ---------------
    The pipeline guarantees that:

        row index i in train.csv / test.csv
        == row index i in y_train / y_test
        == key str(i) in merge_map_train.json / merge_map_test.json

    This function relies *only* on that index alignment; it does not inspect the
    values of y_train / y_test, only their lengths.

    Parameters
    ----------
    x_npz : np.lib.npyio.NpzFile or dict-like
        NPZ object (or dict-like) mapping standardized item names to embedding vectors.
    y_train : np.ndarray
        Label array for the training split. Used for its length (number of rows).
    y_test : np.ndarray
        Label array for the test split. Used for its length (number of rows).
    merge_map_train_path : str or Path
        Path to merge_map_train.json, mapping string indices "0", "1", ... to item names.
    merge_map_test_path : str or Path
        Path to merge_map_test.json, mapping string indices "0", "1", ... to item names.

    Returns
    -------
    X_train : np.ndarray, shape (len(y_train), emb_dim)
    X_test : np.ndarray, shape (len(y_test), emb_dim)
        Feature matrices where row i corresponds to row i in y_train / y_test
        and to row i in train.csv / test.csv.
    """

    # Load the merge maps (index -> standardized item name)
    with open(merge_map_train_path, "r", encoding="utf-8") as f:
        train_map = json.load(f)
    with open(merge_map_test_path, "r", encoding="utf-8") as f:
        test_map = json.load(f)

    # Infer embedding dimension from any one embedding vector
    sample_key = next(iter(x_npz.keys()))
    emb_dim = x_npz[sample_key].shape[0]

    def build_x_matrix(n_rows: int, merge_map: dict[str, str]) -> np.ndarray:
        """Build an (n_rows, emb_dim) matrix using index-based lookup via merge_map."""
        X = np.zeros((n_rows, emb_dim), dtype=float)

        for i in range(n_rows):
            item_name = merge_map.get(str(i))
            if item_name is not None and item_name in x_npz:
                X[i] = x_npz[item_name]
            # If item_name is missing or not in x_npz, X[i] remains zeros.

        return X

    n_train = len(y_train)
    n_test = len(y_test)

    X_train = build_x_matrix(n_train, train_map)
    X_test = build_x_matrix(n_test, test_map)

    return X_train, X_test
