# coded by St.Watermelon

## 학습된 칼만 게인을 가져와서 에이전트를 실행시키는 파일
# 필요한 패키지 임포트
import numpy as np
import math
import gym

env_name = 'Pendulum-v0'
env = gym.make(env_name)

# 칼만 게인 읽어옴
gains = np.loadtxt('./save_weights/kalman_gain.txt', delimiter=" ")

T = gains[-1, 0]
T = np.int(T)

Kt = gains[:, 1:4]
kt = gains[:, -1]

# 실행용 초기 각도 설정
i_ang = 180.0*np.pi/180.0
x0 = np.array([math.cos(i_ang), math.sin(i_ang), 0])
# 원하는 초깃값이 생성될 때까지 환경 초기화 계속
bad_init = True
while bad_init:
    state = env.reset()  # shape of observation from gym (3,)
    x0err = state - x0
    if np.sqrt(x0err.T.dot(x0err)) < 0.1:  # x0=(state_dim,)
        bad_init = False


for time in range(T+1):
    env.render()

    Ktt = np.reshape(Kt[time, :], [1, 3])
    action = Ktt.dot(state) + kt[time] # 행동 계산
    action = np.clip(action, -env.action_space.high[0], env.action_space.high[0])
    ang = math.atan2(state[1], state[0]) # 상태변수로부터 각도 계산


    print('Time: ', time, ', angle: ', ang * 180.0 / np.pi, 'action: ', action)

    state, reward, _, _ = env.step(action)

env.close()