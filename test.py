import torch as T
import torch.nn.functional as F
batch_size = 2
episode_length = 2
input_dim = 3
n_action = 4
states = T.ones((batch_size, episode_length, input_dim))
actions = T.zeros((batch_size, episode_length), dtype=T.int64)
actions = F.one_hot(actions, n_action)
actions = T.concat((T.zeros((batch_size, 1, n_action)), actions[:,:-1,:]), dim=1)
output = T.concat((actions, states), dim=2)

print(output)
print(output.shape)

actions = T.tensor([[4, 0], [0, 1]])
print(T.gather(output, 2, actions.unsqueeze(2)))

states = T.ones((3,4))
dones = T.ones_like(states, dtype=T.bool)

states[dones] = 0
print(states)