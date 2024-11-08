import numpy as np
from scipy.spatial.distance import cosine

def _compute_distance(rep1, rep2, metric='euclidean'):
    if metric == 'cosine':
        return cosine(rep1.flatten(), rep2.flatten())
    elif metric == 'euclidean':
        return np.linalg.norm(rep1 - rep2)

def compute_avg_distances(hidden_states,metric='euclidean',printflag=False):
    avg_dis = 0
    dist_groups = {}
    for key, hidden_state_list in hidden_states.items():
        for i in range(len(hidden_state_list)):
            for j in range(i + 1, len(hidden_state_list)):
                dist = _compute_distance(hidden_state_list[i], hidden_state_list[j], metric='cosine')
                avg_dis += dist
        if printflag:print(f"average distance in group {key} is: {avg_dis/len(hidden_state_list)}")
        dist_groups[key] = (avg_dis / len(hidden_state_list))
        avg_dis = 0  # Reset avg_dis for the next group
    return dist_groups