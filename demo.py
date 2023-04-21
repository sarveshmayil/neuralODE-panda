import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import tkinter as tk

from learning_state_dynamics import *
from panda_neuralODE import *
from visualizers import GIFVisualizer, ImageLabel
from panda_pushing_env import TARGET_POSE_OBSTACLES, BOX_SIZE, PandaPushingEnv

DEVICE = 'cpu'

# Load data: 
collected_data = np.load('collected_data.npy', allow_pickle=True)
validation_data = np.load('validation_data.npy', allow_pickle=True)


# Load models
pushing_absolute_dynamics_model = AbsoluteDynamicsModel(3,3).to(DEVICE)
pushing_absolute_dynamics_model.load_state_dict(torch.load('pushing_absolute_dynamics_model.pt'))

pushing_residual_dynamics_model = ResidualDynamicsModel(3,3).to(DEVICE)
pushing_residual_dynamics_model.load_state_dict(torch.load('pushing_residual_dynamics_model.pt'))

absolute_neuralODE_model = Absolute_ODEnet(3,3,10).to(DEVICE)
absolute_neuralODE_model.load_state_dict(torch.load('neuralODE_absolute_dynamics_model.pt'))

residual_neuralODE_model = Residual_ODEnet(3,3,10).to(DEVICE)
residual_neuralODE_model.load_state_dict(torch.load('neuralODE_residual_dynamics_model_neg1.pt'))


# Evaluate models on validation data
val_dataset = SingleStepDynamicsDataset(np.load('validation_data.npy', allow_pickle=True), device=DEVICE)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset))

num_steps = 4
val_multistep_dataset = MultiStepDynamicsDataset(np.load('validation_data.npy', allow_pickle=True), num_steps=num_steps, device=DEVICE)
val_multistep_loader = torch.utils.data.DataLoader(val_multistep_dataset, batch_size=len(val_multistep_dataset))

pose_loss = SE2PoseLoss(block_width=0.1, block_length=0.1).to(DEVICE)
multistep_pose_loss = MultiStepLoss(pose_loss, discount=1).to(DEVICE)
pose_loss = SingleStepLoss(pose_loss).to(DEVICE)

losses = {
    'Single Step Loss': np.zeros(4),
    'Avg. Multi Step Loss': np.zeros(4)
}

for item in val_loader:
    losses['Single Step Loss'][0] += pose_loss(pushing_absolute_dynamics_model, item['state'], item['action'], item['next_state'])
    losses['Single Step Loss'][1] += pose_loss(pushing_residual_dynamics_model, item['state'], item['action'], item['next_state'])
    losses['Single Step Loss'][2] += pose_loss(absolute_neuralODE_model, item['state'], item['action'], item['next_state'])
    losses['Single Step Loss'][3] += pose_loss(residual_neuralODE_model, item['state'], item['action'], item['next_state'])
    
print("---- Single-step Loss ----")
print(f'Validation loss for absolute           dynamics model is {losses["Single Step Loss"][0]}')
print(f'Validation loss for residual           dynamics model is {losses["Single Step Loss"][1]}')
print(f'Validation loss for absolute neuralODE dynamics model is {losses["Single Step Loss"][2]}')
print(f'Validation loss for residual neuralODE dynamics model is {losses["Single Step Loss"][3]}')

for item in val_multistep_loader:
    losses['Avg. Multi Step Loss'][0] += multistep_pose_loss(pushing_absolute_dynamics_model, item['state'], item['action'], item['next_state'])
    losses['Avg. Multi Step Loss'][1] += multistep_pose_loss(pushing_residual_dynamics_model, item['state'], item['action'], item['next_state'])
    losses['Avg. Multi Step Loss'][2] += multistep_pose_loss(absolute_neuralODE_model, item['state'], item['action'], item['next_state'])
    losses['Avg. Multi Step Loss'][3] += multistep_pose_loss(residual_neuralODE_model, item['state'], item['action'], item['next_state'])

losses['Avg. Multi Step Loss'] /= num_steps

print("\n---- Multi-step Loss ----")
print(f'Validation loss for absolute           dynamics model is {losses["Avg. Multi Step Loss"][0] }')
print(f'Validation loss for residual           dynamics model is {losses["Avg. Multi Step Loss"][1]}')
print(f'Validation loss for absolute neuralODE dynamics model is {losses["Avg. Multi Step Loss"][2]}')
print(f'Validation loss for residual neuralODE dynamics model is {losses["Avg. Multi Step Loss"][3]}')
print()


fig = plt.figure(figsize=(10,5), dpi=400)

models = ("Absolute", "Residual", "Absolute NeuralODE", "Residual NeuralODE")
x = np.arange(4)
width = 0.25  # the width of the bars
multiplier = 0

for attribute, values in losses.items():
    offset = width * multiplier
    rects = plt.bar(x + offset, values, width, label=attribute)
    multiplier += 1

plt.ylabel('Loss')
plt.xticks(x + width, models)
plt.legend(loc='upper left', ncols=2)
plt.ylim(0.0, 0.0004)
# plt.ylim(1e-5, 1e-3)
# plt.yscale("log")

plt.show()



# Control on an obstacle free environment

visualizer = GIFVisualizer()

# set up controller and environment
env = PandaPushingEnv(visualizer=visualizer, render_non_push_motions=False,  include_obstacle=True, camera_heigh=800, camera_width=800, render_every_n_steps=5)
controller = PushingController(env, absolute_neuralODE_model, obstacle_avoidance_pushing_cost_function, num_samples=1000, horizon=20)
env.reset()

state_0 = env.reset()
state = state_0

num_steps_max = 20

for i in tqdm(range(num_steps_max)):
    action = controller.control(state)
    state, reward, done, _ = env.step(action)
    if done:
        break

        
# Evaluate if goal is reached
end_state = env.get_state()
target_state = TARGET_POSE_OBSTACLES
goal_distance = np.linalg.norm(end_state[:2]-target_state[:2]) # evaluate only position, not orientation
goal_reached = goal_distance < BOX_SIZE

print(f'GOAL REACHED: {goal_reached}')

visualizer.get_gif()

root = tk.Tk()
lbl = ImageLabel(root)
lbl.pack()
lbl.load('pushing_visualization.gif')
root.mainloop()