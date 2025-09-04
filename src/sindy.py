import numpy as np
import pysindy as ps
import sklearn as sk
import optimizer as opt

class eSINDy:
    
    def __init__(self, library_functions, features=[], feature_names=None, differentiation_method=None, smoother=None, smoother_kws=None):
        self.library_functions = library_functions
        self.df = differentiation_method if differentiation_method is not None else ps.differentiation.FiniteDifference()
        self.smoother = smoother if smoother is not None else (lambda x: x)
        self.smoother_kws=smoother_kws
        self.coef_median = None
        self.coef_mean = None
        self.coef_list = []
        self.features = features
        self.feature_names = [f'x_{i}' for i in range(len(features))] if feature_names==None else feature_names
        
    def check_fit(self):
        if self.coef_median is None:
            raise ValueError("Model has not been fitted yet.")
        if self.coef_mean is None:
            raise ValueError("Model has not been fitted yet.")
        if len(self.coef_list) == 0:
            raise ValueError("Model has not been fitted yet.")
        
    def print_equations(self, coef='median'):
        
        self.check_fit()

        coef_matrix = None
        if coef == 'mean':
            coef_matrix = self.coef_mean
        elif coef == 'median':
            coef_matrix = self.coef_median
        else:
            print(f"Unknown coef type: {coef}. Use 'mean' or 'median'.")
            return

        for i, name in enumerate(self.feature_names):
            terms = [
                f"{coef_matrix[i, j]:.4f}*{getattr(func, '__name__', repr(func))}"
                for j, func in enumerate(self.library_functions)
                if coef_matrix[i, j] != 0
            ]
            equation = " + ".join(terms) if terms else "0"
            print(f"d{name}/dt = {equation}")
    
    def predict(self, x_test_list):
        self.check_fit()
        
        x_test = np.concatenate(x_test_list)
        Theta = self.library_functions.transform(x_test) 
        dXdt_pred = Theta @ self.coef_median
        
        return dXdt_pred
    
    def score(self, x_test_list, t_test_list):
        self.check_fit()
        
        dXdt_pred = self.predict(x_test_list)
        dXdt_exact = np.concatenate([ps.differentiation.FiniteDifference()._differentiate(x, t) for x, t in zip(x_test_list, t_test_list)])
        
        score = sk.metrics.r2_score(dXdt_exact, dXdt_pred)
        return score
    
    def fit(self, 
            x_train_list, 
            t_train_list, 
            alpha = 1e-12,
            sample_weight = None,
            threshold = 0.5,
            max_iter = 20,
            sample_ensemble = False,
            library_ensemble = False,
            n_ensembles = 100,
            n_subset = 0.5,
            stratified_ensemble = False,
            n_hf = None,
            term_prob = 0.9):
        x_train = np.concatenate([self.smoother(x, **self.smoother_kws) for x in x_train_list])
        y_train = np.concatenate([self.df._differentiate(x, t) for x, t in zip(x_train_list, t_train_list)])
        
        self.library_functions.fit(x_train)
        Theta_full = self.library_functions.transform(x_train)
        
        weights = expand_weights(sample_weight, x_list=x_train_list)
        Theta_full, y_train = rescale_trajectories(Theta_full, y_train, weights)
        
        coef_list = []
        
        for _ in range(n_ensembles if (sample_ensemble or library_ensemble) else 1):
            
            Theta, y = extract_samples(Theta_full, y_train, sample_ensemble, stratified_ensemble, n_hf, n_subset)
            
            library_mask = extract_mask(Theta_full.shape[1], term_prob, library_ensemble)
            Theta = Theta[:, library_mask]
            
            coefs = opt.STLSQ(Theta, y, alpha=alpha, threshold=threshold, max_iter=max_iter)
        
            # Expand back to full coefficient vector if library was reduced
            if library_ensemble:
                full_coef = np.full((Theta_full.shape[1], y.shape[1]), np.nan) 
                full_coef[library_mask, :] = coefs
                coefs = full_coef

            coef_list.append(coefs)
        
        self.coef_list = coef_list
        self.coef_median = np.nanmedian(coef_list, axis=0)
        self.coef_mean = np.nanmean(coef_list, axis=0)
        
        return self
    
        
class eWSINDy(eSINDy):

    def __init__(self,
                library_functions,
                win_length=None,
                stride=None,
                features=None,
                feature_names=None,
                pde=True,
                spatiotemporal_grid=None,
                derivative_order=None,
                K=None,
                H_xt=None):

        # Call parent constructor
        super().__init__(
            library_functions=library_functions,
            features=features or [],
            feature_names=feature_names
        )

        self.pde = pde
        self.spatiotemporal_grid = spatiotemporal_grid
        self.derivative_order = derivative_order
        self.K = K
        self.H_xt = H_xt
        self.win_length = win_length
        self.stride = stride

        # --- Validation ---
        if self.pde:
            if (self.spatiotemporal_grid is None or
                self.derivative_order is None or
                self.K is None or
                self.H_xt is None):
                raise ValueError(
                    "For PDE case, you must provide spatiotemporal_grid, derivative_order, K, and H_xt."
                )
        else:
            if self.win_length is None or self.stride is None:
                raise ValueError(
                    "For ODE (non-PDE) case, you must provide win_length and stride."
                )
            
    def _weak_moments(self, X, t):
        N, d = X.shape
        dt = float(np.mean(np.diff(t)))
        Theta_full = self.library_functions.transform(X)  
        P = Theta_full.shape[1]

        # Indices of window starts
        if self.win_length > N:
            raise ValueError("win_length exceeds signal length")
        starts = np.arange(0, N - self.win_length + 1, self.stride)
        M = len(starts)

        A = np.zeros((M, P))
        B = np.zeros((M, d))

        # Fixed window/test functions (same for every window)
        phi = hann_window(self.win_length)                       # sum(phi) = 1
        phip = window_derivative(phi, dt)                   # derivative wrt time

        for m, s in enumerate(starts):
            e = s + self.win_length
            A[m, :] = (Theta_full[s:e, :] * phi[:, None]).sum(axis=0) * dt
            B[m, :] = - (X[s:e, :] * phip[:, None]).sum(axis=0) * dt

        return A, B, starts   
 
    def assembly(self, x_train_list, t_train_list):
        if not self.pde:
            # ODE case
            A_blocks, B_blocks, win_counts = [], [], []
            X_concat = np.vstack(x_train_list)
            self.library_functions.fit(X_concat)
            for x, t in zip(x_train_list, t_train_list):
                A, B, _ = self._weak_moments(x, t)
                A_blocks.append(A)
                B_blocks.append(B)
                win_counts.append(A.shape[0])

            A_full = np.vstack(A_blocks)
            B_full = np.vstack(B_blocks)

        else:

            weak_features = ps.WeakPDELibrary(
                function_library=self.library_functions,
                derivative_order=self.derivative_order,
                spatiotemporal_grid=self.spatiotemporal_grid,
                include_interaction=True,
                K=self.K,
                H_xt=self.H_xt,
            )

            # Build library and weak moments
            weak_features.fit(x_train_list)  

            A_list = weak_features.transform(x_train_list)
            if isinstance(A_list, list):
                A_full = np.vstack([np.asarray(a) for a in A_list])
            else:
                A_full = np.asarray(A_list)

            # Fix: compute B per trajectory
            B_list = [weak_features.convert_u_dot_integral(x) for x in x_train_list]
            B_full = np.vstack([np.asarray(b) for b in B_list])

            win_counts = [a.shape[0] for a in A_list] if isinstance(A_list, list) else [A_full.shape[0]]
            
            return A_full, B_full, win_counts
    
    def fit(self,
        x_train_list,
        t_train_list,
        sample_weight = None,
        alpha=1e-6,
        threshold=0.1,
        max_iter=20,
        n_ensembles=20,
        sample_ensemble=True,
        library_ensemble=False,
        n_subset=0.5,
        stratified_ensemble=False,
        n_hf=None,
        term_prob=0.9):   
        
        A_full, B_full, win_counts = self.assembly(x_train_list, t_train_list)
        weights = expand_weights(sample_weight, win_counts=win_counts, n_hf=n_hf)
        A_full, B_full = rescale_trajectories(A_full, B_full, weights)
        
        
        coef_list = []
        
        for _ in range(n_ensembles if (sample_ensemble or library_ensemble) else 1):
            
            A, B = extract_samples(A_full, B_full, sample_ensemble, stratified_ensemble, n_hf, n_subset)
            
            library_mask = extract_mask(A.shape[1], term_prob, library_ensemble)
            A = A[:, library_mask]
            
            coefs = opt.STLSQ(A, B, alpha=alpha, threshold=threshold, max_iter=max_iter)
        
            # Expand back to full coefficient vector if library was reduced
            if library_ensemble:
                full_coef = np.full((A.shape[1], B.shape[1]), np.nan) 
                full_coef[library_mask, :] = coefs
                coefs = full_coef

            coef_list.append(coefs)
        
        self.coef_list = coef_list
        self.coef_median = np.nanmedian(coef_list, axis=0)
        self.coef_mean = np.nanmean(coef_list, axis=0)
        
        return self    
                

def expand_weights(weights, 
                   x_list=None, 
                   win_counts=None, 
                   n_hf=None):
    """
    Expand user-provided weights into a full vector.

    ODE case → supply x_list.
    PDE case → supply win_counts and optionally n_hf for stratified HF/LF.
    """
    if weights is None:
        if x_list is not None:
            return np.ones(sum(len(x) for x in x_list))
        if win_counts is not None:
            return np.ones(sum(win_counts))
        return None

    if np.isscalar(weights):
        if x_list is not None:
            return np.full(sum(len(x) for x in x_list), weights)
        if win_counts is not None:
            return np.full(sum(win_counts), weights)

    if isinstance(weights, dict):
        if win_counts is None or n_hf is None:
            raise ValueError("win_counts and n_hf must be provided for dict weights.")
        expanded = []
        for k, n_win in enumerate(win_counts):
            tag = "hf" if k < n_hf else "lf"
            expanded.extend([weights[tag]] * n_win)
        return np.array(expanded)

    weights = np.atleast_1d(weights)

    if x_list is not None and len(weights) == len(x_list):
        return np.concatenate([np.full(len(x), w) for x, w in zip(x_list, weights)])

    if x_list is not None and len(weights) == sum(len(x) for x in x_list):
        return weights

    if win_counts is not None and len(weights) == len(win_counts):
        return np.concatenate([np.full(n_win, w) for w, n_win in zip(weights, win_counts)])

    if win_counts is not None and len(weights) == sum(win_counts):
        return weights

    raise ValueError("Could not expand weights: shape mismatch.")

def rescale_trajectories(X, Y, weights):
    if weights is None:
        return X, Y
    sqrt_w = np.sqrt(weights)
    return X * sqrt_w[:, None], Y * sqrt_w[:, None]
    
def stratified_split(n_total, n_hf, n_subset, seed = None):

    rng = np.random.default_rng(seed)
    n_lf = n_total - n_hf

    # Proportions
    prop_hf = n_hf / n_total
    n_hf_subset = round(n_subset * prop_hf)
    n_lf_subset = n_subset - n_hf_subset  # remainder

    # Draw indices
    idx_hf = rng.choice(n_hf, n_hf_subset, replace=True)
    idx_lf = rng.choice(n_lf, n_lf_subset, replace=True) + n_hf

    # Combine & shuffle
    idx_total = np.concatenate([idx_hf, idx_lf])
    rng.shuffle(idx_total)

    return idx_total
                
def extract_samples(x, y, flag_sample, flag_stratified, n_hf, n_subset):
    if flag_sample:
        if flag_stratified:
            if n_hf == None:
                raise ValueError("n_hf must be a integer >= 1")
            indices = stratified_split(x.shape[0], n_hf, int(x.shape[0]*n_subset))
        else:
            indices = np.random.choice(x.shape[0], size=int(x.shape[0]*n_subset), replace=False)
        x = x[indices]
        y = y[indices]
    else:
        x = x
        y = y
    return x, y
                
def extract_mask(n_features, term_prob, library_flag):
    if library_flag:
        mask = np.random.rand(n_features) < term_prob
        if not np.any(mask):
            mask[np.random.randint(n_features)] = True
        return mask
    return np.ones(n_features, dtype=bool)      
        
def hann_window(n):
    # Discrete Hann window with unit L1 norm
    if n < 3:
        w = np.ones(n)
    else:
        w = 0.5 - 0.5*np.cos(2*np.pi*np.arange(n)/(n-1))
    return w / (np.sum(w) + 1e-15)

def window_derivative(phi, dt):
    # Discrete time derivative of test function, centered finite difference
    # Length preserved by using gradient; scales by dt
    dphi_dt = np.gradient(phi, dt)
    # No normalization; moments carry the correct scaling via dt
    return dphi_dt                