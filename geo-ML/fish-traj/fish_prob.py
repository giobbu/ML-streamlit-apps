import numpy as np
import matplotlib.pyplot as plt
from utils import traj_centre, conc_all_traj, frame2traj, fish_movie_traj, confidence_ellipse 
import matplotlib.animation as animation
import seaborn as sns
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--f', type=int, help='choose experiment based on number of fishes')
parser.add_argument('--n', type=int,  help='choose experiment 1 or 2 or 3')
args = parser.parse_args()

# load experiment zebrafish
frames_dict = np.load('folder_'+str(args.f)+'/'+ str(args.n)+'.npy', allow_pickle=True).item()

# plot movie of experimenti
n_frames = frames_dict.get('trajectories').shape[0]
fish_movie_traj(frames_dict.get('trajectories'), n_frames)

# get centroid over time
stack_centroid = traj_centre(frames_dict.get('trajectories'))

# get centre coordinates
x_centre = stack_centroid[:,0]
y_centre = stack_centroid[:,1]

# get 3-D trajectory
# get time axis
time = [i/2 for i in range(len(stack_centroid))]

# plot traj-1 in 3-D
f = plt.figure()
ax = f.add_subplot(111, projection='3d')
ax.set_xlabel('x-coord')
ax.set_ylabel('y-coord')
ax.set_zlabel('time(frame)')
ax.scatter(x_centre, y_centre, time, s=1)
ax.scatter(x_centre, y_centre, s=1, c='red')
plt.show()






# get trajectories dict for each fish 
n_fish = frames_dict.get('trajectories').shape[1]
fish_dict = frame2traj(frames_dict, n_fish)

# stack all fish trajectories
stack_traj_tot = conc_all_traj(fish_dict, n_fish)

# get x coordinate
x_tot = [stack_traj_tot[i, 0] for i in range(len(stack_traj_tot))]
# get y coordinate
y_tot = [stack_traj_tot[i, 1] for i in range(len(stack_traj_tot))]

# plot hist2d 1000x1000
plt.hist2d(x_tot, y_tot, bins=(1000, 1000), density=True, cmap=plt.cm.jet)
plt.colorbar()
plt.show()

# plot 500x500
plt.hist2d(x_tot, y_tot, bins=(500, 500), density=True, cmap=plt.cm.jet)
plt.colorbar()
plt.show()

# plot 100x100
plt.hist2d(x_tot, y_tot, bins=(100, 100),  density=True, cmap=plt.cm.jet)
plt.colorbar()
plt.show()

# plot 50x50
plt.hist2d(x_tot, y_tot, bins=(50, 50), density=True, cmap=plt.cm.jet)
plt.colorbar()
plt.show()


# Make the plot

# get trajectory fish-0
#traj_0 = np.vstack(fish_dict[0])

# get trajectory fish-1
#traj_1 = np.vstack(fish_dict[1])

# get distance
#dist = traj_0 - traj_1

# get velocity 
#vel_0 = np.diff(traj_0, axis=0) 

# stack all fish trajectories
stack_traj_tot = conc_all_traj(fish_dict, n_fish)

# plot histogram
# get x coordinate
x_tot = [stack_traj_tot[i, 0] for i in range(len(stack_traj_tot))]
# get y coordinate
y_tot = [stack_traj_tot[i, 1] for i in range(len(stack_traj_tot))]

plt.hist2d(x_tot, y_tot, bins=(1000, 1000), cmap=plt.cm.jet)
plt.show()
# plot 500x500
#plt.hist2d(x_tot, y_tot, bins=(500, 500), cmap=plt.cm.jet)
#plt.show()
# plot 100x100
#plt.hist2d(x_tot, y_tot, bins=(100, 100), cmap=plt.cm.jet)
#plt.show()
# plot 50x50
#plt.hist2d(x_tot, y_tot, bins=(50, 50), cmap=plt.cm.jet)
#plt.show()


# plot confidence ellips
from utils import fish_movie, confidence_ellipse # Function to add confidence ellipses to charts
import matplotlib.animation as animation

fig, ax = plt.subplots(figsize = (8, 8))
	
frame_0 = data.get('trajectories')[0]
stack_frame_0 = np.vstack(frame_0) 
x_frame_0 = stack_frame_0[:, 0]
y_frame_0 = stack_frame_0[:, 1]

frame_1 = data.get('trajectories')[1]
stack_frame_1 = np.vstack(frame_1) 
x_frame_1 = stack_frame_1[:, 0]
y_frame_1 = stack_frame_1[:, 1]

# plot movie of experiment
fish_movie(data.get('trajectories'))


# PCA
#from sklearn.decomposition import PCA

# PCA works better if the data is centered
#x = x_frame_1 - np.mean(x_frame_1) # Center x 
#y = y_frame_1 - np.mean(y_frame_1) # Center y
#xy = np.concatenate(([x] , [y]), axis=0).T


#lt.scatter(x_frame_1, y_frame_1)

#pca = PCA(n_components=2)
#pcaTr = pca.fit(xy)

#print('Eigenvectors or principal component: First row must be in the direction of [1, n]')
#print(pcaTr.components_)
#print('Eigenvalues or explained variance')
#print(pcaTr.explained_variance_)

#print('normalized variance')
#print(pcaTr.explained_variance_/sum(pcaTr.explained_variance_))

#norm_variance = pcaTr.explained_variance_/sum(pcaTr.explained_variance_)
# Plot the first component axe. Use the explained variance to scale the vector
#plt.plot([0, pcaTr.components_[0][0] * norm_variance[0]], [0, pcaTr.components_[0][1] * norm_variance[0]], 'k-', color='red')
# Plot the second component axe. Use the explained variance to scale the vector
#plt.plot([0, pcaTr.components_[1][0] * norm_variance[1]], [0, pcaTr.components_[1][1] * norm_variance[1]], 'k-', color='green')
#plt.show()

#plt.scatter(x_frame_0, y_frame_0, s=1)
#confidence_ellipse(x_frame_0, y_frame_0, ax, n_std=1, edgecolor='black', linestyle=':', label=r'$3\sigma$')
#confidence_ellipse(x_frame_0, y_frame_0, ax, n_std=2, edgecolor='black', linestyle=':', label=r'$3\sigma$')
#confidence_ellipse(x_frame_0, y_frame_0, ax, n_std=3, edgecolor='black', linestyle=':', label=r'$3\sigma$')
#plt.show()









# get trajectory fish-0
traj_0 = np.vstack(fish_dict['crazy_fish_0'])
# get x coordinate
x_0 = [traj_0[i, 0] for i in range(len(traj_0))]
# get y coordinate
y_0 = [traj_0[i, 1] for i in range(len(traj_0))]
# plot traj-0
plt.scatter(x_0, y_0, s=1)

# get trajectory fish-1
traj_1 = np.vstack(fish_dict['crazy_fish_1'])
# get x coordinate
x_1 = [traj_1[i, 0] for i in range(len(traj_1))]
# get y coordinate
y_1 = [traj_1[i, 1] for i in range(len(traj_1))]
# plot traj-1
plt.scatter(x_1, y_1, s=1)
plt.show()


# get 3-D trajectory
# get time axis
time = [i/2 for i in range(len(traj_1))]

# plot traj-1 in 3-D
f = plt.figure()
ax = f.add_subplot(111, projection='3d')
ax.set_xlabel('x-coord')
ax.set_ylabel('y-coord')
ax.set_zlabel('time(frame)')
ax.scatter(x_1, y_1, time, s=1)
plt.show()


