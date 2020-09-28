'''
author: junbo hao
date: 2020/9/28
'''

import numpy as np
import matplotlib.pyplot as plt

# 定义迷宫矩阵MAZE
MAZE = [[-1, -1000, -1000, -1],
        [-1, -1, -1000, -1],
        [-1000, -1, -1, -1],
        [-1, -1, -1000, 1000]]
MAZE = np.array(MAZE)


def R_maker(MAZE=MAZE):
	'''
	用于通过迷宫构建R矩阵
	:param MAZE: 迷宫矩阵
	:return: R矩阵
	'''
	[h, w] = np.shape(MAZE)
	num_state = h * w

	# R矩阵为num_state行，4列的矩阵,每行为[上, 下, 左, 右]四个动作的奖励
	R = np.ones([num_state, 4])
	R = -R

	# MAZE矩阵上边缘的state的上动作的奖励定为-1000
	R[0:w, 0] = -1000

	# MAZE矩阵下边缘的state的上动作的奖励定为-1000
	R[-w:, 1] = -1000

	# MAZE矩阵左边缘的state的上动作的奖励定为 - 1000
	R[0::w, 2] = -1000

	# MAZE矩阵右边缘的state的上动作的奖励定为 - 1000
	R[w - 1::w, 3] = -1000

	for i in range(w):
		for j in range(h):
			if MAZE[i, j] == 1000:
				if i < h - 1:  # 处理MAZE矩阵中的下邻居
					R[(i + 1) * w + j, 0] = 1000
				if i > 0:  # 处理MAZE矩阵中的上邻居
					R[(i - 1) * w + j, 1] = 1000
				if j < w - 1:  # 处理MAZE矩阵中的右邻居
					R[i * w + j + 1, 2] = 1000
				if j > 0:  # 处理MAZE矩阵中的左邻居
					R[i * w + j - 1, 3] = 1000

			elif MAZE[i, j] == -1000:
				if i < h - 1:  # 处理MAZE矩阵中的下邻居
					R[(i + 1) * w + j, 0] = -1000
				if i > 0:  # 处理MAZE矩阵中的上邻居
					R[(i - 1) * w + j, 1] = -1000
				if j < w - 1:  # 处理MAZE矩阵中的右邻居
					R[i * w + j + 1, 2] = -1000
				if j > 0:  # 处理MAZE矩阵中的左邻居
					R[i * w + j - 1, 3] = -1000
	return R


R = R_maker(MAZE)

w, h = np.shape(MAZE)

Q = np.zeros([np.size(MAZE), 4])

start_position = 0

reward_list = []

epsilon = 0.1

alpha = 0.1

for episode_index in range(100):
	position = start_position
	MAZE_reward = MAZE[position // w, position % w]
	reward = 0
	step = 0
	while MAZE_reward < 1000:

		# 依概率选择最优动作或随即动作
		if np.random.random() > epsilon:
			action = np.argmax((Q + R)[position, :])
		else:
			wait_actions = R[position, :]
			wait_actions_index = np.argwhere(wait_actions >= -1)
			action_index = np.random.randint(0, np.size(wait_actions_index))
			action = wait_actions_index[action_index]

		# 计算动作收益
		reward = reward + R[position, action]

		# 通过计算动作和当前position得到下一个position
		if action == 0:
			next_position = position - w
		if action == 1:
			next_position = position + w
		if action == 2:
			next_position = position - 1
		if action == 3:
			next_position = position + 1

		# 贝尔曼方程
		new_Q = (1 - alpha) * Q[position, action] + \
		        alpha * (R[position, action] + epsilon * np.max(Q[next_position, :]))

		Q[position, action] = new_Q

		position = next_position

		step = step + 1

		MAZE_reward = MAZE[position // w, position % w]

	print('episode', episode_index, 'step', step)
	reward_list.append(reward)

plt.plot(reward_list)
plt.show()
