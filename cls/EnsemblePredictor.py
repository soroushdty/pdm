# Get the absolute path of the directory above the current working directory
module_path = os.path.abspath(os.path.join('..'))
# Insert the path into sys.path at index 0 to ensure it is checked first
if module_path not in sys.path:
    sys.path.insert(0, module_path)

from cls.MultiLabelLogistic import MultiLabelLogistic

class EnsemblePredictor:
    """ Container for the final ensemble artifact """
    def __init__(self, models, preprocessors, calibrators, thresholds, device='cpu'):
        self.models = models # List of state_dicts
        self.preprocessors = preprocessors
        self.calibrators = calibrators
        self.thresholds = thresholds
        self.device = device

    def predict_proba(self, X):
        all_probs = []
        for i in range(len(self.models)):
            X_trans = self.preprocessors[i].transform(X)
            n_in = self.models[i]['linear.weight'].shape[1]
            n_out = self.models[i]['linear.weight'].shape[0]
            model = MultiLabelLogistic(n_in, n_out)
            model.load_state_dict(self.models[i])
            model.to(self.device)
            model.eval()
            with torch.no_grad():
                X_t = torch.tensor(X_trans, dtype=torch.float32).to(self.device)
                probs = torch.sigmoid(model(X_t)).cpu().numpy()

            class_list = list(self.calibrators[i].keys())
            calibrated = np.zeros_like(probs)
            for c_idx, cls in enumerate(class_list):
                cal = self.calibrators[i][cls]
                col_p = probs[:, c_idx].reshape(-1, 1)
                if hasattr(cal, 'predict_proba'):
                    calibrated[:, c_idx] = cal.predict_proba(col_p)[:, 1]
                else:
                    calibrated[:, c_idx] = cal.predict(col_p).flatten()
            all_probs.append(calibrated)
        return np.mean(np.stack(all_probs), axis=0)
