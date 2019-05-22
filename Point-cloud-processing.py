import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import random
import numpy as np

def run_ransac(data, estimate, is_inlier, sample_size, goal_inliers, max_iterations, stop_at_goal=True, random_seed=None):
    best_ic = 0
    best_model = None
    random.seed(random_seed)
    data = list(data)
    for i in range(max_iterations):
        s = random.sample(data, int(sample_size))
        m = estimate(s)
        ic = 0
        for j in range(len(data)):
            if is_inlier(m, data[j]):
                ic += 1
        if ic > best_ic:
            best_ic = ic
            best_model = m
            if ic > goal_inliers and stop_at_goal:
                break
    print('best model:', best_model)
    return best_model, best_ic

def augment(xyzs):
    axyz = np.ones((len(xyzs), 4))
    axyz[:, :3] = xyzs
    return axyz

def estimate(xyzs):
    axyz = augment(xyzs[:3])
    return np.linalg.svd(axyz)[-1][-1, :]

def is_inlier(coeffs, xyz, threshold):
    return np.abs(coeffs.dot(augment([xyz]).T)) < threshold

fig = plt.figure()
ax = mplot3d.Axes3D(fig)
def plot_plane(a, b, c, d):
    xx, yy = np.mgrid[:10, :10]
    return xx, yy, (-d - a * xx - b * yy) / c

raw = []
coords = []
p = float(input())
n = int(input())
for i in range(n):
    raw.append(input())
    raw[i] = raw[i].split()
    coords.append([float(raw[i][0]), float(raw[i][1]), float(raw[i][2])])
max_iterations = 20
goal_inliers = n * 0.5
#data
xyzs = np.array(coords)
print(xyzs)
ax.scatter3D(xyzs.T[0], xyzs.T[1], xyzs.T[2]) 
# RANSAC
m, b = run_ransac(xyzs, estimate, lambda x, y: is_inlier(x, y, 0.01), 3, goal_inliers, max_iterations)
a, b, c, d = m
xx, yy, zz = plot_plane(a, b, c, d)
ax.plot_surface(xx, yy, zz, color=(0, 0, 0, 0.5))
plt.show()
