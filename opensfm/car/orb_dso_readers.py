import numpy as np

def parse_dso(path_to_dso):
    pos_list = []
    with open(path_to_dso) as f:
        content = f.readlines()
        for item in content:
            i_list = item.split()
            pos = i_list[1:4]
            pos = np.array(pos, dtype=np.float32)
            # print(pos)
            pos_list.append(pos)

    return np.array(pos_list)


def parse_orb(path_to_orb):
    pos_list = []
    with open(path_to_orb) as f:
        content = f.readlines()
        for item in content:
            i_list = item.split()
            pos = i_list[1:4]
            pos = np.array(pos, dtype=np.float32)
            pos_list.append(pos)

    return np.array(pos_list)
