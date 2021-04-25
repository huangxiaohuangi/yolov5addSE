import os

batch_size = 4
world_size = 2
nw = os.cpu_count() // world_size
print(nw)
