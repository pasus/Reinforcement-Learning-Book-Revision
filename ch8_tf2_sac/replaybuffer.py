## 리플레이 버퍼 클래스 파일

# 필요한 패키지 임포트
import numpy as np
from collections import deque
import random

class ReplayBuffer(object):
    """
    Reply Buffer
    """
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque()
        self.count = 0

    ## 버퍼에 저장
    def add_buffer(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)

        # 버퍼가 꽉 찼는지 확인
        if self.count < self.buffer_size:
            self.buffer.append(transition)
            self.count += 1
        else: # 찼으면 가장 오래된 데이터 삭제하고 저장
            self.buffer.popleft()
            self.buffer.append(transition)

    ## 버퍼에서 데이터 무작위로 추출 (배치 샘플링)
    def sample_batch(self, batch_size):
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)
        # 상태, 행동, 보상, 다음 상태별로 정리
        states = np.asarray([i[0] for i in batch])
        actions = np.asarray([i[1] for i in batch])
        rewards = np.asarray([i[2] for i in batch])
        next_states = np.asarray([i[3] for i in batch])
        dones = np.asarray([i[4] for i in batch])
        return states, actions, rewards, next_states, dones


    ## 버퍼 사이즈 계산
    def buffer_count(self):
        return self.count


    ## 버퍼 비움
    def clear_buffer(self):
        self.buffer = deque()
        self.count = 0

