# 演習解答欄

## 演習 1.1.2

作成している行
agent: agent = agents.RandomAgent(env)
env: env = gym.make('EasyMaze-v0')

呼び出されている変数・メソッド

agent:
* agent.act_and_train(obs, reward, done)
* agent.act(obs)
* agent.get_statistics()
* agent.stop_episode_and_train(obs, reward, done)
* agent.stop_episode()
* agent.get_statistics()

env:
* env.metadata.get('render.modes', [])
* env.reset()
* env.render(render_mode)
* env.step(action)

## 演習 1.1.3

平均step数

train: 44.87
test: 44.36

## 演習 1.1.8

平均step数

train: 7.03
test: 6.49

train first 10: 12.7
train last 10: 7.5

## 演習 1.1.9

平均step数

train: 6.66
test: 4.0

## 演習 1.1.10

平均step数

train: 16.18
test: 4.0
