from game.noisy_lever_game import NoisyLeverGame



# Environment parameters
mean_payoffs = [2., 4.]
sigma = 0.5
sigma1 = 1
sigma2 = 2
episode_length = 2


# Initialize environment
env = NoisyLeverGame(mean_payoffs, sigma, sigma1, sigma2, episode_length)

# Reset environment
obs = env.reset()
print(f'obs: {obs} (initial)')

# Step through environment
done = False
while not done:
    action = [0,0]
    obs, reward, done = env.step(action)
    print(f'obs: {obs}, reward: {reward}, done: {done}')