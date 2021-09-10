## 로컬 제어법칙: LQR
# coded by St.Watermelon

# 필요한 패키지 임포트
import numpy as np
import scipy as sp
from config import configuration

## 로컬 제어법칙 구조 정의
class ControlData(object):

    def __init__(self, Kt=None, kt=None, St=None, chol_St=None, inv_St=None, eta=None):

        self.Kt = Kt                # (T+1, action_dim, state_dim)
        self.kt = kt                # (T+1, action_dim)
        self.St = St                # (T+1, action_dim, action_dim)
        # St의 촐레스키 분해
        self.chol_St = chol_St      # (T+1, action_dim, action_dim)
        # St의 역행렬
        self.inv_St = inv_St        # (T+1, action_dim, action_dim)
        self.eta = eta


## 로컬 제어법칙(LQR) 설계
class LocalControl(object):

    def __init__(self, T, state_dim, action_dim):

        self.T = T  # t=0, 1, ..., T
        self.state_dim = state_dim
        self.action_dim = action_dim


    ## 제어법칙 초기화
    def init(self):
        Kt = np.zeros([self.T+1, self.action_dim, self.state_dim])
        kt = np.zeros([self.T+1, self.action_dim])
        St = np.zeros([self.T+1, self.action_dim, self.action_dim])
        chol_St = np.zeros([self.T+1, self.action_dim, self.action_dim])
        inv_St = np.zeros([self.T+1, self.action_dim, self.action_dim])
        T = self.T

        # 칼만 게인은 모두 0으로 초기화, St는 1로 설정
        for t in range(T+1):
            St[t, :, :] = 1.0 * np.eye(self.action_dim)
            inv_St[t,:, :] = 1.0 / St[t, :, :]
            chol_St[t, :, :] = sp.linalg.cholesky(St[t, :, :])
        eta = configuration['init_eta']
        return ControlData(Kt, kt, St, chol_St, inv_St, eta)


    ## 로컬 제어법칙(LQR) 업데이트
    def update(self, control_data, dynamics_data, cost_param, goal_state, eta,
               kl_step_mult, MAX_iLQR_ITER=20):

        T = self.T
        max_eta = configuration['max_eta']
        min_eta = configuration['min_eta']

        # KL 한계(epsilon) 설정
        kl_bound = kl_step_mult * configuration['base_kl_step'] * (T+1)

        for itr in range(MAX_iLQR_ITER):

            # LQR 역방향 패스
            backward_pass = self.backward(control_data, dynamics_data, eta, cost_param, goal_state)

            # LQR 순방향 패스
            xu_mu, xu_cov = self.forward(backward_pass, dynamics_data)
            # KL발산 계산
            kl_div = self.trajectory_kl(xu_mu, xu_cov, backward_pass, control_data)
            constraint = kl_div - kl_bound

            # eta가 kl_bound(epsilon)의 10% 범위 내에 들면 조정을 끝냄
            if abs(constraint) < 0.1 * kl_bound:
                print("KL converged iteration: ", itr)
                break

            # eta 조정
            if constraint < 0:  # eta가 크면
                max_eta = backward_pass.eta
                geo_mean = np.sqrt(min_eta*max_eta) # geometric mean
                new_eta = max(geo_mean, 0.1*max_eta)
            else:   # eta가 작으면
                min_eta = backward_pass.eta
                geo_mean = np.sqrt(min_eta*max_eta)
                new_eta = min(10*min_eta, geo_mean)
            eta = new_eta
        return backward_pass


    ## LQR 역방향 패스
    def backward(self, control_data, dynamics_data, eta, cost_param, goal_state):
        T = self.T
        state_dim = self.state_dim
        action_dim = self.action_dim

        # 모델 파라미터 추출
        fxu = dynamics_data.fxu            # (T+1, state_dim, (state_dim+action_dim))
        fc = dynamics_data.fc              # (T+1, state_dim)

        # 초기화
        Kt = np.zeros((T+1, action_dim, state_dim))
        kt = np.zeros((T+1, action_dim))
        St = np.zeros((T+1, action_dim, action_dim)) # Quut_inv
        chol_St = np.zeros((T+1, action_dim, action_dim))
        Quut = np.zeros((T+1, action_dim, action_dim))

        slice_x = slice(state_dim)
        slice_u = slice(state_dim, state_dim + action_dim)
        eta0 = eta
        inc_eta = 1e-4  # Quu가 역행렬이 존재하지 않을 경우, eta 증가량

        Quupd_err = True  # Quu가 역행렬이 존재하지 않으면
        while Quupd_err:
            Quupd_err = False  # Quu가 역행렬이 존재하면

            # 초기화
            Vtt = np.zeros((T+1, state_dim, state_dim))
            vt = np.zeros((T+1, state_dim))
            Qtt = np.zeros((T+1, state_dim + action_dim, state_dim + action_dim))
            qt = np.zeros((T+1, state_dim + action_dim))

            # 대체 비용 계산
            Ctt, ct = self.augment_cost(control_data, eta, cost_param, goal_state)

            for t in range(T, -1, -1):

                if t == T:
                    Qtt[t] = Ctt[t, :, :]
                    qt[t] = ct[t, :]
                else:
                    Qtt[t] = Ctt[t, :, :] + fxu[t, :, :].T.dot(Vtt[t+1, :, :]).dot(fxu[t, :, :])
                    qt[t] = ct[t, :] + fxu[t, :, :].T.dot(vt[t+1, :] + Vtt[t+1, :, :].dot(fc[t, :]))

                Qtt[t] = 0.5 * (Qtt[t] + Qtt[t].T)

                Quu = Qtt[t, slice_u, slice_u]
                Qux = Qtt[t, slice_u, slice_x]
                Qu = qt[t, slice_u]

                try:
                    # Quu의 촐레스키 분해를 계산
                    U = sp.linalg.cholesky(Quu)
                    L = U.T
                except:
                    # 계산이 안 되면 Quu는 역행렬이 존재하지 않으므로 루프를 빠져나옴
                    Quupd_err = True
                    break

                Quut[t, :, :] = Quu
                # Quut의 역행렬 계산
                Quu_inv = sp.linalg.solve_triangular(
                    U, sp.linalg.solve_triangular(L, np.eye(action_dim), lower=True)
                )
                St[t, :, :] = Quu_inv
                chol_St[t, :, :] = sp.linalg.cholesky(Quu_inv)

                # Kt[t] 계산
                Kt[t, :, :] = -sp.linalg.solve_triangular(
                    U, sp.linalg.solve_triangular(L, Qux, lower=True)
                )
                # kt[t] 계산
                kt[t, :] = -sp.linalg.solve_triangular(
                    U, sp.linalg.solve_triangular(L, Qu, lower=True)
                )

                # 상태가치 함수 계산
                Vtt[t, :, :] = Qtt[t, slice_x, slice_x] - Qux.T.dot(Quu_inv).dot(Qux)
                Vtt[t, :, :] = 0.5 * (Vtt[t, :, :] + Vtt[t, :, :].T)
                vt[t, :] = qt[t, slice_x] - Qux.T.dot(Quu_inv).dot(Qu)

            # Quut의 역행렬이 존재하지 않는다면 eta를 증가
            if Quupd_err:
                eta = eta0 + inc_eta
                inc_eta *= 2.0
                print('Ooops ! Quu is not PD')

                if eta >= 1e16:
                    ValueError('Failed to find PD solution even for very large eta')

        return ControlData(Kt, kt, St, chol_St, Quut, eta)


    ## LQR 순방향 패스
    def forward(self, backward_pass, dynamics_data):
        T = self.T
        state_dim = self.state_dim
        action_dim = self.action_dim

        # 칼만 게인 추출
        Kt, kt, St = backward_pass.Kt, backward_pass.kt, backward_pass.St

        # 모델 파라미터 추출
        fxu = dynamics_data.fxu            # (T+1, state_dim, (state_dim+action_dim))
        fc = dynamics_data.fc              # (T+1, state_dim)
        dyn_cov = dynamics_data.dyn_cov  # (T+1, state_dim, state_dim)

        # 초기화
        xu_mu = np.zeros((T+1, state_dim + action_dim))
        xu_cov = np.zeros((T+1, state_dim + action_dim, state_dim + action_dim))

        slice_x = slice(state_dim)
        xu_mu[0, slice_x] = dynamics_data.x0mu
        xu_cov[0, slice_x, slice_x] = dynamics_data.x0cov

        for t in range(T+1):
            # xu 평균 계산
            xu_mu[t,:] = np.hstack([
                xu_mu[t, slice_x],
                Kt[t,:,:].dot(xu_mu[t, slice_x]) + kt[t, :]
            ])
            # xu 공분산 계산
            xu_cov[t,:,:] = np.vstack([
                np.hstack([
                    xu_cov[t, slice_x, slice_x],
                    xu_cov[t, slice_x, slice_x].dot(Kt[t,:,:].T)
                ]),
                np.hstack([
                    Kt[t,:,:].dot(xu_cov[t, slice_x, slice_x]),
                    Kt[t,:,:].dot(xu_cov[t, slice_x, slice_x]).dot(Kt[t,:,:].T) + St[t,:,:]
                ])
            ])

            if t < T:
                xu_mu[t+1, slice_x] = fxu[t, :, :].dot(xu_mu[t, :]) + fc[t, :]
                xu_cov[t+1, slice_x, slice_x] = fxu[t,:,:].dot(xu_cov[t,:,:]).dot(fxu[t,:,:].T) + dyn_cov[t,:,:]

        return xu_mu, xu_cov


    ## 대체 비용함수 계산
    def augment_cost(self, policy_data, eta, cost_param, goal_state):
        T = self.T
        state_dim = self.state_dim
        action_dim = self.action_dim

        slice_x = slice(state_dim)
        slice_u = slice(state_dim, state_dim + action_dim)

        # 원래 비용함수
        Ctt = np.zeros((state_dim + action_dim, state_dim + action_dim))
        Ctt[slice_x, slice_x] = cost_param['wx'] * 2.0
        Ctt[slice_u, slice_u] = cost_param['wu'] * 2.0
        ct = np.zeros(state_dim + action_dim)
        ct[slice_x] = -2.0 * cost_param['wx'].dot(goal_state)

        # 초기화
        Hessian = np.zeros((T+1, state_dim + action_dim, state_dim + action_dim))
        Jacobian = np.zeros((T+1, state_dim + action_dim))
        Dtt = np.zeros((T+1, state_dim + action_dim, state_dim + action_dim))
        dt = np.zeros((T+1, state_dim + action_dim))

        for t in range(T+1):
            # 이전 제어법칙 칼만 게인 및 공분산 추출
            inv_Sbar = policy_data.inv_St[t,:,:] # (action_dim, state_dim)
            KBar = policy_data.Kt[t, :, :]       # (action_dim, state_dim)
            kbar = policy_data.kt[t, :]          # (action_dim,)
            # 헤시안 계산
            Hessian[t, :, :] = np.vstack([
                np.hstack([KBar.T.dot(inv_Sbar).dot(KBar), -KBar.T.dot(inv_Sbar)]),
                np.hstack([-inv_Sbar.dot(KBar), inv_Sbar])
            ])  # (state_dim+action_dim, state_dim+action_dim)
            # 자코비안 계산
            Jacobian[t, :] = np.concatenate([
                KBar.T.dot(inv_Sbar).dot(kbar), -inv_Sbar.dot(kbar)
            ])   # (state_dim+action_dim,)
            # 대체 비용함수 계산
            Dtt[t,:,:] = Ctt / eta + Hessian[t, :, :]
            dt[t,:] = ct / eta + Jacobian[t, :]

        return Dtt, dt


    ## KL 발산 계산
    def trajectory_kl(self, xu_mu, xu_cov, backward_pass, policy_data):
        T = self.T
        state_dim = self.state_dim
        action_dim = self.action_dim
        slice_x = slice(state_dim)

        # 초기화
        kl_div_t = np.zeros(T+1)

        for t in range(T+1):
            # 이전 제어법칙 칼만 게인 및 공분산 추출
            inv_Sbar = policy_data.inv_St[t, :, :]
            chol_Sbar = policy_data.chol_St[t, :, :]
            KBar= policy_data.Kt[t, :, :]
            kbar = policy_data.kt[t, :]
            # 현재 제어법칙 칼만 게인 및 공분산 추출
            Kt_new = backward_pass.Kt[t, :, :]
            kt_new = backward_pass.kt[t, :]
            St_new = backward_pass.St[t, :, :]
            chol_St_new = backward_pass.chol_St[t, :, :]
            # 칼만 게인 차이
            K_diff = KBar - Kt_new
            k_diff = kbar - kt_new
            # 상태변수 평균 및 공분산
            state_mu = xu_mu[t, slice_x]
            state_cov = xu_cov[t, slice_x, slice_x]
            # 로그_행렬식 (log_determinant)
            logdet_Sbar = 2 * sum(np.log(np.diag(chol_Sbar)))
            logdet_St_new = 2 * sum(np.log(np.diag(chol_St_new)))
            # KL 발산 계산
            kl_div_t[t] = max(
                0,
                0.5 * (
                    np.sum(np.diag(inv_Sbar.dot(St_new))) + \
                    logdet_Sbar - logdet_St_new - action_dim + \
                    k_diff.T.dot(inv_Sbar).dot(k_diff) + \
                    2 * k_diff.T.dot(inv_Sbar).dot(K_diff).dot(state_mu) + \
                    np.sum(np.diag(K_diff.T.dot(inv_Sbar).dot(K_diff).dot(state_cov))) + \
                    state_mu.T.dot(K_diff.T).dot(inv_Sbar).dot(K_diff).dot(state_mu)
                )
            )

        kl_div = np.sum(kl_div_t)
        return kl_div
