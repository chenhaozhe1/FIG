import numpy as np
from numpy.linalg import eigh
import skfda


class FIG:
    """
    Functional Information Geometry (FIG)

    Pipeline:
    1. map data into a higher-dimensional functional feature space
       (Fourier / B-spline / RBF).
    2. Estimate local mean and local covariance via sliding windows.
    3. Perform tangent PCA at each point.
    4. Compute symmetric Mahalanobis-type distances.

    Output:
    - MD: n × n symmetric distance matrix
    """
    def __init__(
        self,
        X,
        window_size=30,
        n_components=3,
        normalization=None,
        lift_type="fourier",
        num_basis=5,
        period=20,
        center_window=None,
        center_stride=None,
        rbf_centers=None,
        rbf_sigma=None,
        random_state=0,
    ):
        """
        Parameters
        ----------
        X : ndarray of shape (n, d)
            Input data matrix, where n is the number of samples (e.g., time points)
            and d is the ambient dimension. Each row corresponds to one observation.

        window_size : int, default=30
            Number of neighboring samples used to estimate local mean and covariance.
            Controls the locality of tangent space estimation: smaller values yield
            more local but noisier geometry, larger values yield smoother geometry.

        n_components : int, default=3
            Number of principal components from the functional PCA.
            Interpreted as the intrinsic dimensionality of the data manifold.

        normalization : {None, "sqrt", "exp"}, default=None
            Normalization applied to tangent-space projections when computing
            distances. "sqrt" corresponds to Mahalanobis-type scaling by the square
            root of eigenvalues, while "exp" exponentially downweights directions
            with large local variance.

        lift_type : {"fourier", "bspline", "rbf"}, default="fourier"
            Type of feature transformation applied prior to geometry estimation.
            "fourier" and "bspline" perform coordinate-wise functional transformation,
            while "rbf" performs multivariate nonlinear transformation.

        num_basis : int, default=5
            Dimensionality of the lifted feature space. For Fourier/B-spline transformation,
            this is the number of basis functions per coordinate; for RBF transformation,
            this is the number of RBF centers.

        period : int or float, default=20
            Period of the Fourier basis. Only used when lift_type="fourier";
            ignored otherwise.

        center_window : int or None, default=None
            Window size for optional averaging of features prior to local
            geometry estimation. Acts as temporal smoothing and noise reduction.

        center_stride : int or None, default=None
            Step size between consecutive centering windows. Must be specified
            together with center_window.

        random_state : int, default=0
            Random seed used for RBF center selection and bandwidth estimation,
            ensuring reproducible geometry and distances.
        """

        self.X = X
        self.n, self.d = X.shape

        self.window_size = window_size
        self.n_components = n_components
        self.normalization = normalization

        self.lift_type = lift_type.lower()
        self.num_basis = num_basis
        self.period = period

        self.center_window = center_window
        self.center_stride = center_stride
        self.random_state = random_state

        # User-controlled RBF hyperparameters
        self.rbf_centers = rbf_centers
        self.rbf_sigma = rbf_sigma

        # Cached internal parameters (for reproducibility)
        self._rbf_centers = None
        self._rbf_sigma = None

        self.MD = None

    # ------------------------------------------------------------------
    # Basis construction (coordinate-wise)
    # ------------------------------------------------------------------

    def _build_basis(self):
        """
        Construct a functional basis object from skfda.
        """
        if self.lift_type == "fourier":
            return skfda.representation.basis.FourierBasis(
                domain_range=(-1, 1),
                n_basis=self.num_basis,
                period=self.period,
            )
        elif self.lift_type == "bspline":
            return skfda.representation.basis.BSpline(
                n_basis=self.num_basis
            )
        else:
            raise ValueError(f"Unknown basis type: {self.lift_type}")

    def lift_features_basis(self):
        """
        Coordinate-wise functional lifting.

        X ∈ R^{n×d} → Φ(X) ∈ R^{n×(d·M)×1}
        """
        basis = self._build_basis()
        lifted = []

        for j in range(self.d):
            xj = self.X[:, j]
            phi_xj = basis(xj).reshape(self.num_basis, self.n).T
            lifted.append(phi_xj)

        lifted_X = np.hstack(lifted)
        return lifted_X[:, :, None]

    # ------------------------------------------------------------------
    # RBF feature lifting (multivariate)
    # ------------------------------------------------------------------

    def lift_features_rbf(self):
        """
        Multivariate RBF lifting.

        X ∈ R^{n×d} → Φ(X) ∈ R^{n×M×1}
        """
        rng = np.random.default_rng(self.random_state)

        # -------- centers --------
        if self.rbf_centers is not None:
            centers = self.rbf_centers
        else:
            if self._rbf_centers is None:
                idx = rng.choice(self.n, size=min(self.num_basis, self.n), replace=False)
                self._rbf_centers = self.X[idx]
            centers = self._rbf_centers

        # cast for speed
        X = self.X.astype(np.float32, copy=False)
        centers = centers.astype(np.float32, copy=False)

        # -------- sigma --------
        if self.rbf_sigma is not None:
            sigma = float(self.rbf_sigma)
        else:
            if self._rbf_sigma is None:
                sample_idx = rng.choice(self.n, size=min(300, self.n), replace=False)
                diffs = X[sample_idx][:, None, :] - centers[None, :, :]
                dists = np.sqrt(np.sum(diffs**2, axis=2))
                self._rbf_sigma = max(np.median(dists[dists > 0]), 1e-6)
            sigma = self._rbf_sigma

        sigma = np.float32(sigma)

        # -------- distance computation (optimized) --------
        X_norm2 = np.sum(X**2, axis=1, keepdims=True)
        C_norm2 = np.sum(centers**2, axis=1, keepdims=True).T
        sq_dists = X_norm2 + C_norm2 - 2 * X @ centers.T
        sq_dists = np.maximum(sq_dists, 0.0)

        Phi = np.exp(-sq_dists / (2 * sigma**2))
        return Phi[:, :, None]
    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    def lift_features(self):
        if self.lift_type == "rbf":
            return self.lift_features_rbf()
        else:
            return self.lift_features_basis()

    # ------------------------------------------------------------------
    # Optional centering (trajectory smoothing)
    # ------------------------------------------------------------------

    def center_features(self, lifted_X):
        """
        Average lifted features over sliding windows.
        This reduces noise and downsamples trajectories.
        """
        if self.center_window is None or self.center_stride is None:
            return lifted_X

        half = int(np.ceil(self.center_window / 2))
        centers = np.arange(half, lifted_X.shape[0], self.center_stride)

        centered = np.zeros((len(centers), lifted_X.shape[1], 1))

        for i, c in enumerate(centers):
            start = max(0, c - half)
            end = min(lifted_X.shape[0], c + half + 1)
            centered[i] = lifted_X[start:end].mean(axis=0)

        return centered

    # ------------------------------------------------------------------
    # Local geometry
    # ------------------------------------------------------------------

    def _sliding_window_neighbors(self, n):
        """
        Construct index-based sliding windows.
        """
        half = self.window_size // 2
        inds = np.zeros((n, self.window_size), dtype=int)

        for i in range(n):
            start = max(0, i - half)
            end = min(n, i + half + 1)
            win = np.arange(start, end)

            if len(win) < self.window_size:
                win = np.pad(win, (0, self.window_size - len(win)), mode="edge")

            inds[i] = win[:self.window_size]

        return inds

    def local_mean(self, data, neighbors):
        """
        Local mean μ_i = average of neighbors around i.
        """
        mean = np.zeros_like(data)
        for i in range(data.shape[0]):
            mean[i, :, 0] = data[neighbors[i], :, 0].mean(axis=0)
        return mean

    def local_covariance(self, data, mean, neighbors):
        """
        Local covariance:
        Σ_i = E[(x - μ_i)(x - μ_i)^T]
        """
        n, d = data.shape[:2]
        cov = np.zeros((n, d, d))

        for i in range(n):
            acc = np.zeros((d, d))
            mu_i = mean[i, :, 0]

            for j in neighbors[i]:
                diff = data[j, :, 0] - mu_i
                acc += np.outer(diff, diff)

            cov[i] = acc / len(neighbors[i])

        return cov

    def tangent_pca(self, cov):
        """
        Eigen-decomposition of local covariance matrices.
        """
        eigvals = np.zeros((cov.shape[0], self.n_components))
        eigvecs = np.zeros((cov.shape[0], self.n_components, cov.shape[1]))

        for i in range(cov.shape[0]):
            w, v = eigh(cov[i])
            idx = np.argsort(w)[::-1][:self.n_components]
            eigvals[i] = w[idx]
            eigvecs[i] = v[:, idx].T

        return eigvals, eigvecs

    # ------------------------------------------------------------------
    # Distances
    # ------------------------------------------------------------------

    def compute_distances(self, data, mean, eigvals, eigvecs):
        """
        Symmetric Mahalanobis-type FIG distance.
        """
        eps = 1e-8
        n = data.shape[0]
        Omega = np.zeros((n, n, self.n_components))

        for i in range(n):
            mu_i = mean[i, :, 0]
            for j in range(n):
                delta = data[j, :, 0] - mu_i
                proj = delta @ eigvecs[i].T

                if self.normalization == "sqrt":
                    proj /= np.sqrt(eigvals[i] + eps)
                elif self.normalization == "exp":
                    proj /= np.exp(-eigvals[i] + 1)

                Omega[i, j] = proj

        MD = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                MD[i, j] = np.sqrt(
                    np.linalg.norm(Omega[i, j])**2 +
                    np.linalg.norm(Omega[j, i])**2
                )

        return MD

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self):
        """
        Run the full FIG pipeline and return distance matrix.
        """
        lifted = self.lift_features()
        lifted = self.center_features(lifted)

        neighbors = self._sliding_window_neighbors(lifted.shape[0])
        mean = self.local_mean(lifted, neighbors)
        cov = self.local_covariance(lifted, mean, neighbors)
        eigvals, eigvecs = self.tangent_pca(cov)

        self.MD = self.compute_distances(lifted, mean, eigvals, eigvecs)
        return self.MD
