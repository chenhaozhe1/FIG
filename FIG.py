import numpy as np
from sklearn.neighbors import NearestNeighbors
from numpy import linalg as LA
import skfda
from phate import PHATE

class FIG:
    def __init__(self, X=None, L=30, n_components=3, normalization=None, num_basis=5, basis_type="Fourier", period=20, L1=None, L3=None):
        self.X = X
        self.L = L
        self.n_components = n_components
        self.normalization = normalization
        self.num_basis = num_basis
        self.basis_type = basis_type
        self.period = period
        self.L1 = L1
        self.L3 = L3
        self.a_vec = self.centering() if L1 is not None and L3 is not None else self.compute_a_vec()
        self.MD = self.fit()

    def compute_KNN(self, k):
        knn = NearestNeighbors(n_neighbors=k + 1)
        knn.fit(self.X)
        knn_inds = knn.kneighbors(self.X, return_distance=False)[:, 1:]
        N = self.X[knn_inds]
        return N, knn_inds

    def win(self, n, L):
        knn_inds = np.zeros((n, L), dtype=int)
        half_L = L // 2
        for i in range(n):
            start = max(0, i - half_L)
            end = min(n, i + half_L)
            inds = np.arange(start, end)
            if len(inds) < L:
                inds = np.pad(inds, (0, L - len(inds)), 'edge')
            knn_inds[i, :] = inds[:L]
        return knn_inds

    def compute_KNN_PHATE(self, k):
        phate_emb_ = PHATE(k=20, n_components=3)
        Y = phate_emb_.fit_transform(self.X)

        knn = NearestNeighbors(n_neighbors=k + 1)
        knn.fit(Y)
        knn_inds = knn.kneighbors(Y, return_distance=False)[:, 1:]
        N = Y[knn_inds]
        return N, knn_inds, Y

    def compute_a_vec(self):
        n, d = self.X.shape
        a_vec = []

        for j in range(d):
            xj = self.X[:, j]

            if self.basis_type == "BSpline":
                basis = skfda.representation.basis.BSpline(n_basis=self.num_basis)
            elif self.basis_type == "Fourier":
                basis = skfda.representation.basis.FourierBasis(
                    domain_range=(-20, 20),
                    n_basis=self.num_basis,
                    period=self.period
                )
            phi = basis(xj).reshape(self.num_basis, n).T
            a_vec.append(phi)

        return np.hstack(a_vec).reshape(n, -1, 1)

    def centering(self):
        n_obs = self.X.shape[0]
        centers = np.arange(np.ceil(self.L1 / 2), n_obs + np.ceil(self.L1 / 2), self.L3)
        n = centers.shape[0]
        features = self.compute_a_vec()
        a_vec_centers = np.zeros((n, features.shape[1], 1))

        for i in range(n):
            c = centers[i]
            phi_t = features[int(c - np.ceil(L1/2)): int(c + np.ceil(L1/2)),:]
            mu = np.mean(phi_t, axis = 0)
            a_vec_centers[i, :] = mu

        return a_vec_centers

    def compute_data_vec(self):
        n, d = self.X.shape
        data_vec = np.zeros((n, d, 1))
        for i in range(n):
            xi = self.X[i, :]
            data_vec[i, :, 0] = xi.reshape(d, 1)
        return data_vec

    def compute_mean(self, data, knn_inds):
        n, d = data.shape[:2]
        k = knn_inds.shape[1]
        mean_vec = np.zeros((n, d, 1))

        for i in range(n):
            neighbors = data[knn_inds[i]].reshape(k, d)
            mean_vec[i, :, 0] = np.mean(neighbors, axis=0)

        return mean_vec

    def compute_A_mat(self, data, mu, knn_inds):
        n, d = data.shape[:2]
        k = knn_inds.shape[1]
        A = np.zeros((n, d, d))

        for i in range(n):
            cum = np.zeros((d, d))
            for j in range(k):
                idx = knn_inds[i, j]
                a_j = data[idx, :, 0]
                mu_j = mu[idx, :, 0]
                cum += np.outer(a_j, a_j) - np.outer(mu_j, mu_j)
            A[i, :, :] = cum / (2 * k)

        return A

    def compute_PCs(self, A):
        n, dim = A.shape[:2]
        EigVal = np.zeros((n, self.n_components))
        EigVec = np.zeros((n, self.n_components, dim))

        for i in range(n):
            Ai = A[i, :, :]
            eigvals, eigvecs = LA.eig(Ai)
            indices = np.argsort(eigvals)[::-1]
            eigvals, eigvecs = eigvals[indices], eigvecs[:, indices]
            EigVal[i, :] = eigvals[:self.n_components]
            EigVec[i, :, :] = eigvecs[:, :self.n_components].T

        return EigVal, EigVec

    def compute_projections(self, data, mu, EigVec, EigVal):
        n = data.shape[0]
        Omega = np.zeros((n, n, self.n_components))

        for i in range(n):
            mu_i = mu[i, :, 0]
            for j in range(n):
                a_j = data[j, :, 0]
                theta_ijk = (a_j - mu_i).T @ EigVec[i, :, :].T
                if self.normalization == 'sqrt':
                    w_ijk = theta_ijk / np.sqrt(EigVal[i, :])
                elif self.normalization == 'exp':
                    w_ijk = theta_ijk / np.exp(-EigVal[i, :] + 1)
                else:
                    w_ijk = theta_ijk
                Omega[i, j, :] = w_ijk

        return Omega

    def fit(self):
        n = self.a_vec.shape[0]
        knn_inds = self.win(n=n, L=self.L)
        mu_vec = self.compute_mean(data=self.a_vec, knn_inds=knn_inds)
        A = self.compute_A_mat(data=self.a_vec, mu=mu_vec, knn_inds=knn_inds)
        EigVal, EigVec = self.compute_PCs(A=A)
        Omega = self.compute_projections(data=self.a_vec, mu=mu_vec, EigVec=EigVec, EigVal=EigVal)

        MD = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist1 = np.linalg.norm(Omega[i, i, :] - Omega[i, j, :])
                dist2 = np.linalg.norm(Omega[j, i, :] - Omega[j, j, :])
                MD[i, j] = np.sqrt(dist1 ** 2 + dist2 ** 2)

        return MD

    def add_noise(self, sigma):
        n, d = self.X.shape
        mean = np.zeros(d)
        noise = np.random.multivariate_normal(mean, (sigma**2) * np.identity(d), n)
        X_noise = self.X + noise
        return X_noise

