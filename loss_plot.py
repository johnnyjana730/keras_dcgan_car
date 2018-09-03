import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import spline
import os
SCRIPT_PATH = os.path.dirname(os.path.abspath( __file__ ))

#Plotting loss
with open(SCRIPT_PATH +'/train_record/training_log.txt') as f:
    log_data = f.readlines()
    

train_steps = []
disc_real_loss = []
disc_fake_loss = []
GAN_loss = []
for log in log_data:
    train_steps.append(int(log.split(' ')[1]))
    disc_real_loss.append(float(log.split(' ')[5]))
    disc_fake_loss.append(float(log.split(' ')[8]))
    GAN_loss.append(float(log.split(' ')[11][0:-1]))

#real vs fake loss plot
fig = plt.figure()
ax = fig.add_subplot(111)

lns1 = ax.plot(train_steps, disc_real_loss, '-', label="Discriminator real loss")
lns2 = ax.plot(train_steps, disc_fake_loss, '-', label="Discriminator fake loss")

lns = lns1+lns2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0)

ax.grid()
ax.set_xlabel("Train steps")
ax.set_ylabel("Loss")
# plt.show()
plt.savefig(SCRIPT_PATH +'/train_record/real_vs_fake_loss_plot.png')

#gen loss plot
fig = plt.figure()
ax = fig.add_subplot(111)

lns3 = ax.plot(train_steps, GAN_loss, '-', label="Generator loss")
lns = lns3
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0)

ax.grid()
ax.set_xlabel("Train steps")
ax.set_ylabel("Loss")
# plt.show()
plt.savefig(SCRIPT_PATH +'/train_record/gen_loss_plot.png')