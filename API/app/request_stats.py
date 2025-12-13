import numpy as np

def get_stats(values):

    if not values:
        return {
            'count': 0,
            'mean': 0,
            'median': 0, 
            'q95': 0, 
            'q99': 0
            } 

    cnt = len(values)
    mean_stat = np.mean(values)
    q50, q95, q99 = np.quantile(values, [0.5, 0.95, 0.99])

    stats_dict = {
        'count': cnt,
        'mean': np.round(mean_stat, 3),
        'median': np.round(q50, 3), 
        'q95': np.round(q95, 3), 
        'q99': np.round(q99, 3)
    }

    return stats_dict