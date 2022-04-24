import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib.animation as animation
import numpy as np

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`
   """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
                      width=ell_radius_x * 2,
                      height=ell_radius_y * 2,
                      facecolor=facecolor,
                      **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


# get centroid points 
def traj_centre(trajectories):
	centroid_l = []
	for i in range(trajectories.shape[0]):
		centre = trajectories[i].mean(axis=0)
		centroid_l.append(centre)
	stack_centroid = np.vstack(centroid_l)
	return stack_centroid


# plot experiment movie of zebrafish
def fish_movie_traj(trajectories, n_frames):
    # First set up the frame coordinates
    frame = trajectories[1]
    centre = np.array(frame.mean(axis=0))
    stack_frame = np.vstack(frame) 
    x_frame = stack_frame[:, 0]
    y_frame = stack_frame[:, 1]

    # animation function.  This is called sequentially
    def animate(i):
        frame = trajectories[i]
        centre = np.array(frame.mean(axis=0))
        sc.set_offsets(frame)
        sc_centre.set_offsets(centre)
        #line.set_data(centre[0], centre[1])
        return sc, sc_centre, #line

    fig, ax = plt.subplots(figsize=(5,5)) 
    sc = ax.scatter(x_frame, y_frame, color ='green')
    sc_centre = ax.scatter(centre[0], centre[1], color ='red')
    #line = ax.plot([], [])
    # call the animator
    anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=10)
    return plt.show()


def frame2traj(dict_frame, n_fish):
	fish_dict = {key:[] for key in range(n_fish)}
	frame_l = dict_frame.get('trajectories')
	for frame in frame_l:
		for i, row in enumerate(frame):
			fish_dict[i].append(row)
	return fish_dict


def conc_all_traj(fish_dict, n_fish):	
	# get probability
	traj_tot_l = []
	# concatenate all trajectories across all frames
	for i in range(n_fish):
		traj_i = np.vstack(fish_dict[i])
		traj_tot_l.append(traj_i)
	stack_traj_tot = np.vstack(traj_tot_l)
	# check for NaNs values
	print('NaN Values:' + str(np.isnan(np.sum(stack_traj_tot))))
	stack = stack_traj_tot[~np.isnan(stack_traj_tot).any(axis=1), :]
	return stack





