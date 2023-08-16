# A2C load and play (tf2 version)
# coded by St.Watermelon

## 학습된 신경망 파라미터를 가져와서 에이전트를 실행시키는 파일
# 필요한 패키지 임포트
import gymnasium as gym
import tensorflow as tf
from a2c_learn import A2Cagent

def main():

    env_name = 'Pendulum-v1'
    env = gym.make(env_name, render_mode = "human")

    agent = A2Cagent(env)

    agent.load_weights('./save_weights/')  # 신경망 파라미터를 가져옴

    time = 0
    state = env.reset()[0] # 환경을 초기화하고 초기 상태 관측

    while True:
        env.render()

        action = agent.actor(tf.convert_to_tensor([state], dtype=tf.float32))[0][0] # 행동 계산
        state, reward, done, _trunc, _info = env.step(action)  # 환경으로 부터 다음 상태, 보상 받음
        time += 1

        print('Time: ', time, 'Reward: ', reward)

        if done:
            break

    env.close()

if __name__=="__main__":
    main()