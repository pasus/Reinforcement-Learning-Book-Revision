# DDPG main (tf2 subclassing API version)
# coded by St.Watermelon
## DDPG 에이전트를 학습하고 결과를 도시하는 파일

# 필요한 패키지 임포트
import gym
from ddpg_learn import DDPGagent

def main():

    max_episode_num = 200  # 최대 에피소드 설정
    env = gym.make("Pendulum-v0")
    agent = DDPGagent(env)  # DDPG 에이전트 객체

    # 학습 진행
    agent.train(max_episode_num)

    # 학습 결과 도시
    agent.plot_result()


if __name__=="__main__":
    main()