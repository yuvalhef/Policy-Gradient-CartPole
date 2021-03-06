import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os


env = gym.make('CartPole-v1')

np.random.seed(1)


class PolicyNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='policy_network'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):

            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.placeholder(tf.int32, [self.action_size], name="action")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")

            self.W1 = tf.get_variable("W1", [self.state_size, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b1 = tf.get_variable("b1", [8], initializer=tf.zeros_initializer())
            self.W2 = tf.get_variable("W2", [8, self.action_size], initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b2 = tf.get_variable("b2", [self.action_size], initializer=tf.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.sigmoid(self.Z1)
            self.output = tf.add(tf.matmul(self.A1, self.W2), self.b2)

            # Softmax probability distribution over actions
            self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output))
            # Loss with negative log probability
            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.action)
            self.loss = tf.reduce_mean(self.neg_log_prob * self.R_t)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


class ValueEstimator:
    def __init__(self, state_size, learning_rate, name='value_network'):
        self.state_size = state_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):

            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state_v")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards_v")

            self.W1 = tf.get_variable("W1", [self.state_size, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b1 = tf.get_variable("b1", [8], initializer=tf.zeros_initializer())
            self.W2 = tf.get_variable("W2", [8, 1], initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b2 = tf.get_variable("b2", [1], initializer=tf.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.sigmoid(self.Z1)
            self.output = tf.add(tf.matmul(self.A1, self.W2), self.b2)

            # Softmax probability distribution over actions
            self.value_estimate = tf.squeeze(self.output)
            self.loss = tf.squared_difference(self.value_estimate, self.R_t)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)


# Define hyperparameters
state_size = 4
action_size = env.action_space.n

max_episodes = 5000
max_steps = 501
discount_factor = 0.99
learning_rate = 0.0009

render = False

# Initialize the policy network
tf.reset_default_graph()
policy = PolicyNetwork(state_size, action_size, learning_rate)
value = ValueEstimator(state_size, 0.006)


# Start training the agent with REINFORCE algorithm
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    solved = False
    episode_rewards = np.zeros(max_episodes)
    average_rewards = 0.0
    rewards_stats = []
    losses_v = []
    losses_e = []
    first = False

    for episode in range(max_episodes):
        state = env.reset()
        state = state.reshape([1, state_size])
        episode_transitions = []

        for step in range(max_steps):
            actions_distribution = sess.run(policy.actions_distribution, {policy.state: state})
            action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.reshape([1, state_size])

            if render:
                env.render()

            action_one_hot = np.zeros(action_size)
            action_one_hot[action] = 1
            episode_rewards[episode] += reward
            action = action_one_hot

            # Calculate TD Target
            value_next = sess.run(value.value_estimate, {value.state: next_state})
            state_value = sess.run(value.value_estimate, {value.state: state})
            if done:
                td_target = reward
                td_error = td_target - state_value
            else:
                td_target = reward + discount_factor * value_next
                td_error = td_target - state_value

            # Update the value estimator
            feed_dict_v = {value.state: state, value.R_t: td_target}
            _, loss_value = sess.run([value.optimizer, value.loss], feed_dict_v)
            losses_v.append(loss_value)

            # Update the policy estimator
            # using the td error as our advantage estimate
            feed_dict = {policy.state: state, policy.R_t: td_error, policy.action: action}
            _, loss = sess.run([policy.optimizer, policy.loss], feed_dict)
            losses_e.append(loss)

            if done:
                # if np.mean(episode_rewards[(episode - 9):episode+1]) > 400:
                #     value.learning_rate *= 0.99
                #     policy.learning_rate *= 0.99
                #     print(policy.learning_rate)
                #     first = True
                # else:
                #     if first:
                #         value.learning_rate *= 1.001
                #         policy.learning_rate *= 1.001
                #         print(policy.learning_rate)

                if episode > 98:
                    # Check if solved
                    average_rewards = np.mean(episode_rewards[(episode - 99):episode+1])
                    rewards_stats.append(average_rewards)
                print("Episode {} Reward: {} Average over 100 episodes: {}".format(episode, episode_rewards[episode], round(average_rewards, 2)))
                if average_rewards > 475:
                    print(' Solved at episode: ' + str(episode))
                    solved = True
                break
            state = next_state

        if solved:
            break


def plot_reward(dir, start, rewards, name):
    df = pd.DataFrame()
    df[name] = rewards
    df['episode'] = list(range(start, len(rewards)+start))
    ax =df.plot(x='episode', y=name)
    ax.set_xlabel("Episode")
    ax.set_ylabel(name)
    ax.legend().remove()
    ax = plt.savefig(dir+'/'+name+'.jpg')


def plot_loss(dir, name, losses):
    ax = pd.DataFrame(losses).plot()
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss - "+name+' network')
    ax.legend().remove()
    ax = plt.savefig(dir+'/Loss_'+name+'.jpg')


log_dir = 'actor_critic'
os.makedirs(log_dir, exist_ok=True)

plot_reward(log_dir, 100, rewards_stats, 'Average 100 Episode Rewards')
plot_reward(log_dir, 0, episode_rewards[:episode], 'Episode Rewards')
plot_loss(log_dir, 'value', losses_v)
plot_loss(log_dir, 'policy', losses_e)
print('max episode: {}'.format(episode))
