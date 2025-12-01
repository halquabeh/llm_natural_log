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


from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
def compute_group_structure_score(pc1, group_labels, group_sizes):
    """
    Compute the Group Structure Score (GSS) for a given set of PC1 values and group labels.
    
    Args:
        pc1 (np.array): PC1 values of shape (n_samples,).
        group_labels (list): Group labels for each sample.
        group_sizes (list): Sizes of the groups.
    
    Returns:
        GSS (float): Group Structure Score.
        MS (float): Monotonicity Score.
        VS (float): Variance Score.
    """
    # Step 1: Compute group means
    group_means = []
    for group in np.unique(group_labels):
        group_means.append(np.mean(pc1[group_labels == group]))
    
    # Step 2: Compute Monotonicity Score (MS)
    ms, _ = spearmanr(group_sizes, group_means)
    
    # Step 3: Compute Variance Score (VS)
    group_variances = []
    for group in np.unique(group_labels):
        group_variances.append(np.var(pc1[group_labels == group]))
    
    # Fit linear regression of log(variance) vs. log(group size)
    X = np.log(group_sizes).reshape(-1, 1)
    y = np.log(group_variances)
    reg = LinearRegression().fit(X, y)
    vs = reg.coef_[0]  # Slope of the regression line
    
    # Step 4: Compute Group Structure Score (GSS)
    w1, w2 = 0.5, 0.5  # Weights for MS and VS
    gss = w1 * ms + w2 * vs
    
    return gss, ms, vs


from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
 

def transform_hidden_states(all_layers_hidden_states, method="PCA", num_components=2):
    use_pls = method.lower() == "pls"
    transformed_hidden_states = {}

    for layer, hidden_states in all_layers_hidden_states.items():
        all_representations = []
        all_answers = []

        # Collect all representations and answers across groups
        group_indices = {}  # Keep track of indices per group to restore structure later
        index = 0
        for group, reps in hidden_states['hidden_states'].items():
            group_indices[group] = list(range(index, index + len(reps)))
            for rep, answer in zip(reps, hidden_states['answers'][group]):
                all_representations.append(np.array(rep))  # Convert tensor to NumPy array
                all_answers.append(answer)  # Keep answers as they are
            index += len(reps)

        all_representations = np.array(all_representations).squeeze(1)  # Remove unnecessary dimensions
        all_answers = np.array(all_answers, dtype=float)

        try:
            if use_pls:
                model = PLSRegression(n_components=num_components)
                model.fit(all_representations, all_answers)
                principal_components = model.transform(all_representations)
                Variances = model.score(all_representations, all_answers)  # R² score
                # print(f'Layer {layer} - PLS R²: {Variances:.3f}')
                first_direction = model.x_weights_[:, 0]  # First direction of PLS
            else:
                model = PCA(n_components=num_components)
                principal_components = model.fit_transform(all_representations)
                Variances = model.explained_variance_ratio_.sum()  # Explained variance
                # print(f'Layer {layer} - PCA Explained Variance: {Variances:.3f}')
                first_direction = model.components_[0]  # First direction of PCA

            # Restore the group structure
            reduced_hidden_states = {}
            for group, indices in group_indices.items():
                reduced_hidden_states[group] = [principal_components[i].tolist() for i in indices]

            # Save the transformed hidden states, transformation metric, and first direction
            transformed_hidden_states[layer] = {
                'hidden_states': reduced_hidden_states,
                'answers': hidden_states['answers'],
                'Explained_variance': Variances,  # Either R² or explained variance
                'First_direction': first_direction.tolist()  # First direction of PCA or PLS
            }

        except ValueError as e:
            print(f'Layer {layer} encountered an issue: {e}')
            continue

    return transformed_hidden_states



import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

def compute_monotonicity(x, transformed_x):
    """Compute the Monotonicity Metric (M²) using Spearman rank correlation."""
    # Convert inputs to lists if they aren't already
    x = list(x) if not isinstance(x, list) else x
    transformed_x = list(transformed_x) if not isinstance(transformed_x, list) else transformed_x
    
    assert isinstance(x, list), "x must be a list"
    assert isinstance(transformed_x, list), "transformed_x must be a list"
    assert len(x) == len(transformed_x), "Input lists must have same length"
    spearman_corr, _ = spearmanr(x, transformed_x)
    return spearman_corr  #in range [-1,1]

def compute_sublinearity(hidden_states, epsilon=1e-8):

    group_means = []
    sorted_groups = sorted(hidden_states['hidden_states'].keys(), key=lambda g: int(g))  # Sort numerically

    for group in sorted_groups:
        group_values = hidden_states['hidden_states'][group]
        group_means.append(np.mean(group_values))  # Take the mean representation for each group

    # Compute consecutive differences
    group_diffs = np.diff(group_means)
    group_diffs = np.abs(group_diffs)  # Ensure positive differences

    # Avoid log(0) or log(negative) by adding a small epsilon
    group_diffs = np.clip(group_diffs, epsilon, None)

    if len(group_diffs) < 2:
        return np.nan  # Not enough points for regression

    # Log-log regression to estimate β
    log_indices = (np.arange(1, len(group_diffs) + 1)).reshape(-1, 1)
    log_diffs = np.log(group_diffs).reshape(-1, 1)

    regress = LinearRegression()
    regress.fit(log_indices, log_diffs)
    logbeta = regress.coef_[0][0]  # Extract slope
    beta = np.exp(logbeta)  # Calculate beta from logbeta
    return beta

def analyze_transformed_hidden_states(transformed_hidden_states):
    idx = 0
    for layer, hidden_states in transformed_hidden_states.items():
        all_representations = []
        all_answers = []
        idx+=1

        # Collect all representations and answers across groups
        for group in hidden_states['hidden_states'].keys():
            for rep, answer in zip(hidden_states['hidden_states'][group], hidden_states['answers'][group]):
                all_representations.append(rep)
                all_answers.append(answer)

        all_representations = np.array(all_representations).flatten()  # Ensure 1D
        

        all_answers = np.array(all_answers, dtype=float)
        # Compute metrics
        # if idx==2:
        #     print(all_answers,all_representations)
        # INSERT_YOUR_CODE
        assert not np.isnan(all_answers).any(), "answers contains NaN values"
        M2 = compute_monotonicity(all_answers, all_representations)
        SM = compute_sublinearity(hidden_states)

        # print(f'Layer {layer} -EV: {hidden_states["Explained_variance"]},  M²: {M2:.3f}, SM: {SM:.3f}')

        # Store metrics in the dictionary
        transformed_hidden_states[layer]['monotonicity_metric'] = M2
        transformed_hidden_states[layer]['sublinearity_metric'] = SM

    return transformed_hidden_states


import random
# def create_random_mapping():
#     # Generate a list of numbers from 0 to 25
#     numbers = list(range(26))
#     # Shuffle the list to get a random permutation
#     random.shuffle(numbers)
#     # Create a dictionary to map each character to its corresponding number
#     mapping ={'a': 23, 'b': 10, 'c': 19, 'd': 8, 'e': 20, 'f': 17, 'g': 4, 'h': 5, 'i': 1, 'j': 9, 'k': 18, 'l': 3, 'm': 24, 'n': 13, 'o': 0, 'p': 11, 'q': 25, 'r': 16, 's': 14, 't': 15, 'u': 6, 'v': 22, 'w': 12, 'x': 7, 'y': 2, 'z': 21}
#     return mapping
# def get_numerical_value(char, mapping):
#     # Convert the character to lowercase to handle both uppercase and lowercase inputs
#     char = char.lower()
#     # Return the corresponding numerical value from the mapping
#     return mapping.get(char, None)
# Example usage:
# mapping = create_random_mapping()
# print("Random Mapping:", mapping)
# Test the function with some characters
# # characters = ['a', 'b', 'c', 'z']
# for char in characters:
#     value = get_numerical_value(char, mapping)
#     print(f"The numerical value of '{char}' is {value}")



