# ------------------------------------------------------------------------------
#
#       LQR_FLM for pendulum-v0
#       data collection type [X, U]
#       state = [x, y, theta_dot]
#       coded by St.Watermelon
# ------------------------------------------------------------------------------

## LQR-FLM (LQR with Fited Linear Model) 에이전트
# 필요한 패키지 임포트
import copy
import numpy as np
import math

from sample_trajectory import TrainingData, Sampler
from linear_dynamics import DynamicsData, LocalDynamics
from gaussian_control import ControlData, LocalControl

from gmm.dynamics_prior_gmm import DynamicsPriorGMM

from config import configuration


class LQRFLMagent(object):

    def __init__(self, env):

        # 환경
        self.env = env
        # 상태변수 차원
        self.state_dim = env.observation_space.shape[0]
        # 행동 차원
        self.action_dim = env.action_space.shape[0]
        # 행동의 최대 크기
        self.action_bound = env.action_space.high[0]

        # GMM 초기
        self.prior = DynamicsPriorGMM()

        # 목표 상태변수
        goal_ang = 0 * np.pi / 180.0
        self.goal_state = np.array([math.cos(goal_ang), math.sin(goal_ang), 0])
        # 초기 상태변수
        i_ang = -45.0 * np.pi / 180.0
        self.init_state = np.array([math.cos(i_ang), math.sin(i_ang), 0])
        # 생성할 궤적 개수와 시간 구간
        self.N = configuration['num_trajectory']
        self.T = configuration['T']

        # 비용함수
        self.cost_param = {
            'wx': np.diag([10.0, 0.01, 0.1]),  # (state_dim, state_dim)
            'wu': np.diag([0.001]),  # (action_dim, action_dim)
        }

        # epsilon
        self.kl_step_mult = configuration['init_kl_step_mult']

        # 궤적 구조 생성
        self.training_data = TrainingData()
        self.prev_training_data = TrainingData()
        self.sampler = Sampler(self.env, self.N, self.T, self.state_dim, self.action_dim)

        # 로컬 동역학 모델 생성 p(x_t+1|xt,ut)
        self.dynamics_data = DynamicsData()
        self.prev_dynamics_data = DynamicsData()
        self.local_dynamics = LocalDynamics(self.T, self.state_dim, self.action_dim, self.prior)

        # 로컬 제어법칙(LQR) 생성 p(ut|xt)
        self.control_data = ControlData()
        self.prev_control_data = ControlData()
        self.local_controller = LocalControl(self.T, self.state_dim, self.action_dim)

        # 에피소드에서 얻은 비용을 저장하기 위한 변수
        self.save_costs = []


    ## 학습
    def update(self, MAX_ITER):

        print("Now, regular iteration starts ...")

        for iter in range(int(MAX_ITER)):

            print("\niter = ", iter)

            # step 1: 이전 로컬 제어법칙으로 궤적 생성
            x0 = self.init_state
            if iter == 0:
                self.control_data = self.local_controller.init()
                self.training_data = self.sampler.generate(x0, self.control_data, self.cost_param, self.goal_state)
            else:
                self.training_data = self.sampler.generate(x0, self.prev_control_data, self.cost_param, self.goal_state)

            # 비용계산
            iter_cost = self.training_data.cost
            self.save_costs.append(iter_cost)
            print("     iter_cost  = ", iter_cost)

            # step 2: 모델 피팅
            self.dynamics_data = self.local_dynamics.update(self.training_data)

            # step 3: 로컬 제어법칙 업데이트
            if iter > 0:
                eta = self.prev_control_data.eta
                self.control_data = self.local_controller.update(self.prev_control_data,
                                                                self.dynamics_data, self.cost_param,
                                                                self.goal_state,
                                                                eta, self.kl_step_mult)

            # step 4: KL step (epsilon) 조정
            if iter > 0:
                self._epsilon_adjust()

            # step 5: 다음 이터레이션 준비
            self._update_iteration_variables()

        # 비용 저장
        np.savetxt('./save_weights/pendulum_iter_cost.txt', self.save_costs)


    ## KL step (epsilon) 조정
    def _epsilon_adjust(self):
        # 이전/현재의 실제 비용
        _last_cost = self.prev_training_data.cost
        _cur_cost = self.training_data.cost
        # 비용 추정
        _expected_cost = self.estimate_cost(self.control_data, self.dynamics_data)

        # 비용 감소량 계산
        _expected_impr = _last_cost - _expected_cost
        _actual_impr = _last_cost - _cur_cost

        print("  cost last, expected, current = ", _last_cost, _expected_cost, _cur_cost)

        # epsilon multiplier 조정
        _mult = _expected_impr / (2.0 * max(1e-4, _expected_impr - _actual_impr))
        _mult = max(0.1, min(5.0, _mult))
        new_step = max(
                    min(_mult * self.kl_step_mult, configuration['max_kl_step_mult']),
                        configuration['min_kl_step_mult']
                    )

        self.kl_step_mult = new_step
        print(" epsilon_mult = ", new_step)


    ## 현재 이터레이션 파라미터를 이전 이터레이션 파라미터로 복사
    def _update_iteration_variables(self):

        self.prev_training_data = copy.deepcopy(self.training_data)
        self.prev_dynamics_data = copy.deepcopy(self.dynamics_data)
        self.prev_control_data = copy.deepcopy(self.control_data)

        self.training_data = TrainingData()
        self.dynamics_data = DynamicsData()
        self.control_data = ControlData()


    ## 비용 추정
    def estimate_cost(self, control_data, dynamics_data):
        T, state_dim, action_dim = self.T, self.state_dim, self.action_dim

        slice_x = slice(state_dim)
        slice_u = slice(state_dim, state_dim + action_dim)

        # 원래 비용
        Ctt = np.zeros((state_dim + action_dim, state_dim + action_dim))
        Ctt[slice_x, slice_x] = self.cost_param['wx'] * 2.0
        Ctt[slice_u, slice_u] = self.cost_param['wu'] * 2.0
        ct = np.zeros((state_dim + action_dim))
        ct[slice_x] = -2.0 * self.cost_param['wx'].dot(self.goal_state)
        cc = self.goal_state.T.dot(self.cost_param['wx']).dot(self.goal_state)

        # 모델 파라미터 추출
        fxu = dynamics_data.fxu            # (T+1, state_dim, (state_dim+action_dim))
        fc = dynamics_data.fc              # (T+1, state_dim)
        dyn_cov = dynamics_data.dyn_cov  # (T+1, state_dim, state_dim)

        # 칼만 게인 및 공분산 추출
        Kt = control_data.Kt
        kt = control_data.kt
        St = control_data.St

        # 초기화
        predicted_cost = np.zeros(T+1)
        xu_mu = np.zeros((T+1, state_dim + action_dim))
        xu_cov = np.zeros((T+1, state_dim + action_dim, state_dim + action_dim))
        xu_mu[0, slice_x] = dynamics_data.x0mu
        xu_cov[0, slice_x, slice_x] = dynamics_data.x0cov

        for t in range(T+1):
            # xu 평균
            xu_mu[t,:] = np.hstack([
                xu_mu[t, slice_x],
                Kt[t,:,:].dot(xu_mu[t, slice_x]) + kt[t, :]
            ])
            # xu 공분산
            xu_cov[t,:,:] = np.vstack([
                np.hstack([
                    xu_cov[t, slice_x, slice_x], xu_cov[t, slice_x, slice_x].dot(Kt[t,:,:].T)
                ]),
                np.hstack([
                    Kt[t,:,:].dot(xu_cov[t, slice_x, slice_x]),
                    Kt[t,:,:].dot(xu_cov[t, slice_x, slice_x]).dot(Kt[t,:,:].T) + St[t,:,:]
                ])
            ])

            if t < T:
                xu_mu[t+1, slice_x] = fxu[t, :, :].dot(xu_mu[t, :]) + fc[t, :]
                xu_cov[t+1, slice_x, slice_x] = fxu[t,:,:].dot(xu_cov[t,:,:]).dot(fxu[t,:,:].T) + dyn_cov[t,:,:]

        for t in range(T+1):
            x = xu_mu[t, slice_x]
            u = xu_mu[t, slice_u]
            # 비용 추정
            predicted_cost[t] = (x - self.goal_state).T.dot(self.cost_param['wx']).dot(x - self.goal_state) + \
                           u.T.dot(self.cost_param['wu']).dot(u) * np.sum(xu_cov[t, :, :]*Ctt)

        return predicted_cost.sum()

