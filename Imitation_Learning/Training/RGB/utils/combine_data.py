import glob
import os
import numpy as np

# matches = []
# for root, dirnames, filenames in os.walk(rootdir):
#     for filename in fnmatch.filter(filenames, pattern):
#         matches.append(os.path.join(root, filename))
#
save_dir = "/mnt/7A0C2F9B0C2F5185/eslam/CARLA_DATA_combined/"
root_dir ="/mnt/7A0C2F9B0C2F5185/eslam/CARLA_DATA/"
data = {}
data[("right", "depth")] = []
data[("right", "seg")] = []
data[("right", "rgb")] = []
data[("right", "labels")] = []
data[("right", "_Thickness")] = []
data[("right", "_PGM")] = []

data[("left", "depth")] = []
data[("left", "seg")] = []
data[("left", "rgb")] = []
data[("left", "labels")] = []
data[("left", "_Thickness")] = []
data[("left", "_PGM")] = []

data[("straight", "depth")] = []
data[("straight", "seg")] = []
data[("straight", "rgb")] = []
data[("straight", "labels")] = []
data[("straight", "_Thickness")] = []
data[("straight", "_PGM")] = []

directions = ["straight","right","left"]
sensors = ["labels", "_PGM", "_Thickness", "depth", "seg", "rgb"]  # "labels","processed","depth", "seg",
# sensors = [ "labels","_PGM"]#,"_Thickness","depth", "seg","rgb"]#"labels","processed","depth", "seg",
paths = []
for root, directories, filenames in sorted(
        os.walk(root_dir)):
    for filename in filenames:
        if ('LiDAR' in filename):
            if ('_Thickness' not in filename and '_PGM' not in filename):
                continue
        paths.append(os.path.join(root, filename))

paths = sorted(paths)
cnt = {}
for i in range(len(paths)):
    print(paths[i][:])
#     if "straight" in paths[i]:
#         idx = paths[i].split('/')[7][8:]
#         print(idx)
#         if idx in cnt:
#             cnt[idx] += 1
#         else:
#             cnt[idx] = 1
# print ('##############################################')
# for i,v in cnt.items():
#     if v!=6:
#         print(i)
# exit(0)
for filename in paths:
    for i in directions:
        for j in sensors:
            if i in filename and j in filename:
                data[(i, j)].append(filename)
# print('RIGHT')
# for i in range(len(data[("right", "rgb")])):
#     print('##########################################################################')
#     print(data[("right", "rgb")][i],'\n', data[("right", "labels")][i])
#     if data[("right", "rgb")][i][1:].replace('rgb', 'labels') == data[("right", "labels")][i]:
#         print('OMGAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD')
#     print('##########################################################################')
# print('LEFT')
# for i in range(len(data[("left", "rgb")])):
#     print('##########################################################################')
#     print(data[("left", "rgb")][i],'\n', data[("left", "labels")][i])
#     if data[("left", "rgb")][i][1:].replace('rgb', 'labels') == data[("left", "labels")][i]:
#         print('OMGAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD')
#     print('##########################################################################')
# print('STRAIGHT')
# for i in range(len(data[("straight", "rgb")])):
#     print('##########################################################################')
#     print(data[("straight", "rgb")][i],'\n', data[("straight", "labels")][i])
#     if data[("straight", "rgb")][i][1:].replace('rgb', 'labels') == data[("straight", "labels")][i]:
#         print('OMGAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD')
#     print('##########################################################################')

# exit(0)

print('##########################################################################')
print('##########################################################################')
# exit(0)
# print(data)
# print('##########################################################################')
# print('##########################################################################')

# exit(0)
for i in directions:
    for j in sensors:
        for item in range(len(data[(i, j)])):
            # print('Path: ', data[(i, j)][item])
            if item == 0:
                all = np.load(data[(i, j)][item])
                continue
            elif ():
                continue
            else:
                # if (i == 'left' and item == 175): ## left lidar is more than expected so upto this number it's even.
                #    break
                temp = np.load(data[(i, j)][item])
            all = np.concatenate((all, temp), axis=0)
            print(i, "_", j, item, '----', all.shape)

        np.save(os.path.join(save_dir, i, j + ".npy"), all)
        print("###################################################")
        print(i, "_", j, " saved", '----', all.shape)
# print(all)
