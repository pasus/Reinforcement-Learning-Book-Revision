# coded by St.Watermelon

## LQR-FLM 에이전트를 학습하고 결과를 도시하는 파일
# 필요한 패키지 임포트
import gym
from lqrflm_agent import LQRFLMagent
import math
import numpy as np
from config import configuration


def main():

    MAX_ITER = 60  # 학습 이터레이션 설정
    env_name = 'Pendulum-v0'  # 환경으로 OpenAI Gym의 pendulum-v0 설정
    env = gym.make(env_name)
    agent = LQRFLMagent(env)  # LQR-FLM 에이전트 객체

    # 학습 진행
    agent.update(MAX_ITER)
    T = configuration['T']
    # 학습된 칼만 게인 추출
    Kt = agent.prev_control_data.Kt
    kt = agent.prev_control_data.kt

    print("\n\n Now play ................")
    # 초기 상태변수 설정
    x0 = agent.init_state

    play_iter = 5
    save_gain = []
    # 학습 결과 플레이
    for pn in range(play_iter):

        print("     play number :", pn+1)

        if pn < 2:
            bad_init = True
            while bad_init:
                state = env.reset()  # shape of observation from gym (3,)
                x0err = state - x0
                if np.sqrt(x0err.T.dot(x0err)) < 0.1:  # x0=(state_dim,)
                    bad_init = False
        else:
            state = env.reset()

        for time in range(T+1):
            env.render()
            # 행동 계산
            action = Kt[time, :, :].dot(state) + kt[time, :]
            action = np.clip(action, -agent.action_bound, agent.action_bound)
            ang = math.atan2(state[1], state[0])

            print('Time: ', time, ', angle: ', ang*180.0/np.pi, 'action: ', action)

            save_gain.append([time, Kt[time, 0, 0], Kt[time, 0, 1], Kt[time, 0, 2], kt[time, 0]])

            state, reward, _, _ = env.step(action)
    # 칼만 게인 저장
    np.savetxt('./save_weights/kalman_gain.txt', save_gain)


if __name__=="__main__":
    main()