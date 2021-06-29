import imageio
import numpy
from fluid_sim import Fluid

FRAMES = 30

flu = Fluid()

video = numpy.full((FRAMES, flu.size, flu.size), 0, dtype=float)

for step in range(0, FRAMES):
    flu.density[4:7, 4:7] += 100  # add density into a 3*3 square
    flu.velo[5, 5] += [1, 2]  # add velocity (direction of flow)

    flu.step()
    video[step] = flu.density

imageio.mimsave('./video.gif', video.astype('uint8'))
