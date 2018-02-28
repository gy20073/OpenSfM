import sys
sys.path.append("../../")
from opensfm import dataset
from opensfm import features
import numpy as np


def return_longest_recon(reconstruction):
    maxi = 0

    def num_images(recon):
        return len(recon.shots.keys())

    for i, recon in enumerate(reconstruction):
        if num_images(recon) > num_images(reconstruction[maxi]):
            maxi = i
    return reconstruction[maxi]

def read_in_data(data_path, mask_path):
    data = dataset.DataSet(data_path)
    reconstruction = data.load_reconstruction()
    reconstruction = [return_longest_recon(reconstruction)]
    graph = data.load_tracks_graph()

    # first read in all images
    mask_dict={}
    for track_id in reconstruction[0].points:
        for imageid in graph[track_id]:
            if imageid not in mask_dict:
                mask_dict[imageid] = data.seg_as_array(imageid, mask_path)

    return (reconstruction, graph, mask_dict)
    
def check_a_point(point, mask):
    # assert that the point is one dimensional
    points = np.array(point)
    points = points[np.newaxis, 0:2]
    points = features.denormalized_image_coordinates(points, mask.shape[1], mask.shape[0])
    p = points[0, :]
    if mask[int(p[1]), int(p[0])] == 0:
        # then not include this point
        return 0
    else:
        # include this point
        return 1
    
def check_3D_point(track_id, masks, graph):
    # return whether want to include this 3D point    
    each_point = []
    for image_id in graph[track_id]:
        loc = graph[track_id][image_id]['feature']
        each_point.append(check_a_point(loc, masks[image_id]))
    # we apply a heuristic of if larger than 1/3 on moving object, then throw it away
    each_point = np.array(each_point)
    
    # 1 means static
    ratio = 1.0 * np.sum(each_point) / len(each_point)
    #print "ratio", ratio
    if ratio < 2.0 / 3:
        return 0 # do not include this point in evaluation
    else:
        return 1 # include this point in the evaluation
    
# for each of the point, determine whether it should be included in the final evaluation
def evaluate_all_points(reconstruction, masks, graph):
    if len(reconstruction) != 1:
        print "reconstruction has more than one segments or 0 segments"
        return 1e9
    else:
        rec = reconstruction[0]

    errors = []
    for track_id in rec.points:
        if check_3D_point(track_id, masks, graph):
            errors.append(rec.points[track_id].reprojection_error)
    return np.mean(errors)

def evaluate(data_path, mask_path):
    reconstruction, graph, mask_dict = read_in_data(data_path, mask_path)

    return evaluate_all_points(reconstruction, mask_dict, graph)
    
if __name__ == "__main__":
    print evaluate(sys.argv[1], sys.argv[2])
    