## 동역학 모델 피팅을 위한 GMM 프라이어
# 필요한 패키지 임포트
import logging
import numpy as np
from gmm.gmm import GMM
from config import configuration

LOGGER = logging.getLogger(__name__)


class DynamicsPriorGMM(object):
    """
    GMM 업데이트 및 상태천이 데이터([x_t, u_t, x_t+1])의 프라이어 값 계산
    다음 논문 참고:
    S. Levine*, C. Finn*, T. Darrell, P. Abbeel, "End-to-end
    training of Deep Visuomotor Policies", arXiv:1504.00702,
    Appendix A.3
    """
    def __init__(self):
        """
        하이퍼파라미터:
            min_samples_per_cluster: 클러스터 당 최소 샘플 궤적 개수.
            max_clusters: 최대 클러스터 개수
            max_samples: 최대 샘플 궤적 개수
            strength: 프라이어 값 조정 인자
        """

        self.X = None
        self.U = None

        self.gmm = GMM()

        self._min_samp = configuration['gmm_min_samples_per_cluster']
        self._max_samples = configuration['gmm_max_samples']
        self._max_clusters = configuration['gmm_max_clusters']
        self._strength = configuration['gmm_prior_strength']


    def initial_state(self):
        """ 초기 상태변수의 프라이어 값 계산"""
        # 평균과 공분산 계산
        mu0 = np.mean(self.X[:, 0, :], axis=0)
        Phi = np.diag(np.var(self.X[:, 0, :], axis=0))

        # n0, m 설정
        n0 = 1.0
        m = 1.0
        # Phi 값 조정
        Phi = Phi * m
        return mu0, Phi, m, n0


    def update(self, training_data):
        """
        GMM 업데이트
        """

        X = training_data.X
        U = training_data.U

        # 상수
        T = X.shape[1] - 1

        # 데이터세트에 상태천이 데이터 추가
        if self.X is None or self.U is None:
            self.X = X
            self.U = U
        else:
            self.X = np.concatenate([self.X, X], axis=0)
            self.U = np.concatenate([self.U, U], axis=0)

        # 데이터세트에서 일정 샘플 개수를 유지
        start = max(0, self.X.shape[0] - self._max_samples + 1)
        self.X = self.X[start:, :]
        self.U = self.U[start:, :]

        # 클러스터 차원 계산
        Do = X.shape[2] + U.shape[2] + X.shape[2]

        # 데이터세트 생성
        N = self.X.shape[0]
        xux = np.reshape(
            np.c_[self.X[:, :T, :], self.U[:, :T, :], self.X[:, 1:(T+1), :]],
            [T * N, Do]
        )

        # 클러스터 개수 선정
        K = int(max(2, min(self._max_clusters,
                           np.floor(float(N * T) / self._min_samp))))
        LOGGER.debug('Generating %d clusters for dynamics GMM.', K)

        # GMM 업데이트
        self.gmm.update(xux, K)


    def eval(self, Dx, Du, pts):
        """
        특정 시간스텝에서 프라이어 값 계산
        """
        # 상태천이 데이터 차원 확인
        assert pts.shape[1] == Dx + Du + Dx

        #  프라이어 값 계산
        mu0, Phi, m, n0 = self.gmm.inference(pts)

        # n0, m 설정
        n0 = 1.0
        m = 1.0

        # Phi 값 조정
        Phi *= m
        return mu0, Phi, m, n0