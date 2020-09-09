from abc import ABC

import gym
import matplotlib.pyplot as plt
import torch
import random

import torch.nn as nn
import torch.nn.functional as func
import torchvision.transforms as trans
from PIL import Image
import numpy as np

env = gym.make('CartPole-v0').unwrapped

plt.ion()

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print(f"{device=}")


class Transition:
    def __init__(self, state, action, next_state, reward):
        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward

    def __str__(self):
        return f'{self.state=} {self.action=} {self.next_state=} {self.reward=}'

    def __repr__(self):
        return str(self)


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Add a batch of transitions to memory"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = Transition(args[0], args[1], args[2], args[3])
        self.position = self.position + 1
        if self.position == self.capacity:
            self.position = 0

    def sample(self, batch_size):
        """Draw batch of size batch_size from memory"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


t = Transition(0, 'run', 1, 100)
print(t)

m = ReplayMemory(10)
m.push(0, 'run', 1, 100)
m.push(1, 'jump', 2, 200)
m.push(2, 'duck', 3, 300)

print(m.sample(2))


class DeepQNetwork(nn.Module, ABC):
    def __init___(self, h, w, outputs):
        super().__init__()
        self.layers = [
            (nn.Conv2d(3, 16, kernel_size=5, stride=2), nn.BatchNorm2d(16)),
            (nn.Conv2d(16, 32, kernel_size=5, stride=2), nn.BatchNorm2d(32)),
            (nn.Conv2d(32, 32, kernel_size=5, stride=2), nn.BatchNorm2d(32)),
        ]

        def conv2d_size_output(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        conv_w = conv2d_size_output(conv2d_size_output(conv2d_size_output(w)))
        conv_h = conv2d_size_output(conv2d_size_output(conv2d_size_output(h)))
        linear_input_size = conv_w * conv_h * 32

        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        for (conv, batch_norm) in self.layers:
            x = func.relu(batch_norm(conv(x)))
        return self.head(x.view(x.size(0), -1))


# Resize image to 40 and return tensor
resize = trans.Compose([
    trans.ToPILImage(),
    trans.Resize(40, interpolation=Image.CUBIC),
    trans.ToTensor()
])


def get_cart_location(screen_width):
    """Return the middle of the cart"""
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)


def get_screen():
    """
    Get screen from gym
    Chop it to the cart
    Convert it to torch tensor
    """

    # Returned screen requested by gym is 400x600x3, but sometimes larger such as 800x1200x3.
    # Transpose it into torch order (CHW).

    screen = env.render(mode='rgb_array').transpose((2, 0, 1))

    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_h, screen_w = screen.shape
    screen = screen[:, int(screen_h * 0.4):int(screen_h * 0.8)]
    view_w = int(screen_w * 0.6)
    cart_location = get_cart_location(screen_w)
    if cart_location < view_w // 2:
        slice_range = slice(view_w)
    elif cart_location > (screen_w - view_w // 2):
        slice_range = slice(view_w)
    else:
        slice_range = slice(cart_location - view_w // 2,
                            cart_location + view_w // 2)

    # Center a square around the cart
    screen = screen[:, :, slice_range]
    # Convert tp float, rescale by 255, convert to torch tensor
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)

    # Resize and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)


env.reset()
plt.figure()
plt.imshow(get_screen().squeeze(0).permute(1, 2, 0).numpy(), interpolation='none')
plt.title('Example of an extracted screen')
plt.show()

env.close()
