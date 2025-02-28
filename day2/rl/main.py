import gym
import gym_easymaze
import agents
from print_buffer import PrintBuffer


# 環境と agent を用意
# env = gym.make('EasyMaze-v0')
env = gym.make('CartPole-v0')
# agent = agents.RandomAgent(env)
# agent = agents.RulebaseAgent(env)
# agent = agents.TableQAgent(env)
agent = agents.DQNAgent(env)


# 描画設定
# train時・test時に各 step を描画するかどうか
prints_detail = {'train': False, 'test': False}
# 何 episode ごとに統計情報を出力するか
every_print_statistics = {'train': 10, 'test': 10}
# 描画モード
# human: 動画をウィンドウで出す ansi: 文字列をstrで返す
supported_render_modes = env.metadata.get('render.modes', [])
render_mode = 'human' if 'human' in supported_render_modes else 'ansi'
# 描画バッファの用意
render_buffer = PrintBuffer()

# 学習設定

# 行うepisode 数
n_episode = {'train': 10000, 'test': 1}

# 何step 経ったらepisode を強制終了するか
n_max_time = {'train': 500, 'test': 500}

# 演習3.4.3
n_steps_sum = {'train': [], 'test': []}

# ゲームを繰り返す
for interact_mode in ['train', 'test']:  # 一周目: train, 二周目: test
    sum_of_all_rewards = 0.0  # episode ごとの average reward を出すために、総計を覚えておく
    for i_episode in range(n_episode[interact_mode]):  # 各episode について
        render_buffer.prints('-------------------------------')
        render_buffer.prints('episode: {0} / {1}'.format(i_episode, n_episode[interact_mode]))
        
#         print(agent.q_table_to_str()) # 演習3.4.8 Qテーブルをエピソードごとに出力
        
        # 環境を初期化して新たな episode を開始する
        reward, done, sum_of_rewards = 0.0, False, 0.0
        obs = env.reset()
        for time in range(n_max_time[interact_mode]):  # 各 step について
            if prints_detail[interact_mode]:
                # 環境を描画
                if render_mode == 'human':
                    env.render(render_mode)
                else:
                    render_buffer.prints(env.render(render_mode))
            # agentに観測から行動を選択してもらう
            if interact_mode == 'train':
                # 学習もしてもらう
                action = agent.act_and_train(obs, reward, done)
            else:
                action = agent.act(obs)
            # 環境に行動を与えて1ステップ進ませる
            obs, reward, done, info = env.step(action)
            
            sum_of_rewards += reward
            render_buffer.prints(interact_mode, 'episode:', i_episode, 'T:', time,
                                 'R:', sum_of_rewards, 'statistics:', agent.get_statistics())
            if prints_detail[interact_mode]:
                render_buffer.flush()  # 表示
            else:
                render_buffer.clear()  # 表示しない
            if done:
                n_steps_sum[interact_mode].append(time) # 演習3.4.3
                # ゲーム終了時(ゲームクリア) の処理
                if prints_detail[interact_mode]:
                    print('Episode finished.')
                break
        
        # ゲーム終了時(時間切れ) の処理
        if prints_detail[interact_mode] and not done:
            n_steps_sum[interact_mode].append(n_max_time[interact_mode]) # 演習3.4.3
            print('{0} steps have past, but the agent could not reach the goal.'.format(time))
        
        # episode が終了したことを agent に伝える
        if interact_mode == 'train':
            # 最後に学習もしてもらう
            agent.stop_episode_and_train(obs, reward, done)
        else:
            agent.stop_episode()
        sum_of_all_rewards += sum_of_rewards
        
        # 数 episodes に一回、統計情報を表示
        if (i_episode + 1) % every_print_statistics[interact_mode] == 0 or prints_detail[interact_mode]:
            average_rewards = sum_of_all_rewards / (i_episode + 1)
            print(interact_mode, 'episode:', i_episode + 1, 'T:', '???',
                  'R:', average_rewards, 'statistics:', agent.get_statistics())
    
    print("step / episode : ")
    print("train : ", str(sum(n_steps_sum["train"])/n_episode["train"]), "first 10 steps:", str(sum(n_steps_sum["train"][:10])/10) , "last 10 steps:", str(sum(n_steps_sum["train"][-10:])/10))
    print("test : ", str(sum(n_steps_sum["test"])/n_episode["test"]), "first 10 steps:", str(sum(n_steps_sum["test"][:10])/10) , "last 10 steps:", str(sum(n_steps_sum["test"][-10:])/10))
    print(interact_mode, 'finished.')
