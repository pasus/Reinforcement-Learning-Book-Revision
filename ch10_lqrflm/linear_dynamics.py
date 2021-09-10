## 로컬 선형 동역학 모델
# coded by St.Watermelon

# 필요한 패키지 임포트
import numpy as np


## 로컬 선형 동역학 모델 구조 정의
class DynamicsData(object):

    def __init__(self, fxu=None, fc=None, dyn_cov=None, x0mu=None, x0cov=None):

        self.fxu = fxu              # (T+1, state_dim, state_dim + action_dim)
        self.fc = fc                # (T+1, state_dim)
        self.dyn_cov = dyn_cov      # (T+1, state_dim, state_dim)

        self.x0mu = x0mu
        self.x0cov = x0cov


## 로컬 선형 동역학 모델 피팅
class LocalDynamics(object):

    def __init__(self, T, state_dim, action_dim, prior):

        self.T = T
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.prior = prior


    ## 선형 모델 업데이트
    def update(self, training_data):

        X = training_data.X
        U = training_data.U
        N = X.shape[0]

        # 프라이어 업데이트
        self.prior.update(training_data)

        # 모델 피팅
        fxu, fc, dyn_cov = self.fit(X, U)

        # 초기 상태변수 및 공분산 추정
        x0 = X[:, 0, :]
        x0mu = np.mean(x0, axis=0)                 # (state_dim,)
        x0cov = np.diag(np.maximum(np.var(x0, axis=0), 1e-6))

        mu00, Phi0, priorm, n0 = self.prior.initial_state()
        x0cov += Phi0 + (N*priorm) / (N+priorm) * np.outer(x0mu-mu00, x0mu-mu00) / (N+n0)

        return DynamicsData(fxu, fc, dyn_cov, x0mu, x0cov)


    ## 모델 피팅
    def fit(self, X, U, cov_reg=1e-6):

        # 초기화
        N = X.shape[0]
        fxu = np.zeros([self.T+1, self.state_dim, self.state_dim + self.action_dim])
        fc = np.zeros([self.T+1, self.state_dim])
        dyn_cov = np.zeros([self.T+1, self.state_dim, self.state_dim])

        slice_xu = slice(self.state_dim + self.action_dim)
        slice_xux = slice(self.state_dim + self.action_dim, self.state_dim + self.action_dim + self.state_dim)

        # 가중치
        dwts = (1.0 / N) * np.ones(N)

        for t in range(self.T+1):

            # xux = [xt;  ut;  x_t+1]
            xux = np.c_[X[:, t, :], U[:, t, :], X[:, t+1, :]] # (N, state_dim+action_dim+state_dim)
            # Normal-inverse-Wishart prior 계산
            mu0, Phi, mm, n0 = self.prior.eval(self.state_dim, self.action_dim, xux)

            # 가중행렬
            D = np.diag(dwts)

            # 평균과 공분산 계산
            xux_mean = np.mean((xux.T * dwts).T, axis=0)
            diff = xux - xux_mean
            xux_cov = diff.T.dot(D).dot(diff)
            xux_cov = 0.5 * (xux_cov + xux_cov.T)

            # MAP 추정
            map_cov = (Phi + N * xux_cov + (N * mm) / (N + mm) * np.outer(xux_mean-mu0, xux_mean-mu0)) / (N + n0)
            map_cov = 0.5 * (map_cov + map_cov.T)
            map_cov[slice_xu, slice_xu] += cov_reg * np.eye(self.state_dim+self.action_dim) # for matrix inverse

            map_mean = (mm * mu0 + n0 * xux_mean) / (mm + n0)

            # 모델 파라미터 추정
            fxut = np.linalg.solve(map_cov[slice_xu, slice_xu], map_cov[slice_xu, slice_xux]).T  # (state_dim, state_dim+action_dim)
            fct = map_mean[slice_xux] - fxut.dot(map_mean[slice_xu]) # (state_dim,)

            proc_cov = map_cov[slice_xux, slice_xux] - fxut.dot(map_cov[slice_xu, slice_xu]).dot(fxut.T)

            fxu[t, :, :] = fxut
            fc[t, :] = fct
            dyn_cov[t, :, :] = 0.5 * (proc_cov + proc_cov.T)

        return fxu, fc, dyn_cov
