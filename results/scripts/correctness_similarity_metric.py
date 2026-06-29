import sys
import os
import numpy as np
import json
import pylcs
import argparse

def edit_dist(hyp, ref):
    tmp = pylcs.edit_distance(hyp, ref)
    res_norm = 1-(tmp/max(len(hyp),len(ref)))
    return res_norm
 
def calc_ed(hyps, refs):
    scores = [edit_dist(h, r) for h, r in zip(hyps, refs)]
    mean_ed = np.mean(scores)
    min_ed = np.min(scores)
    max_ed = np.max(scores)
    median_ed = np.median(scores)
    q1_ed = np.percentile(scores, 25)
    q3_ed = np.percentile(scores, 75)
    formatted_score = (f'ED: {mean_ed * 100:.2f}% (min: {min_ed:.3f}, max: {max_ed:.3f}, median: {median_ed:.3f}, Q1: {q1_ed:.3f}, Q3: {q3_ed:.3f})')
    print(formatted_score)
    return formatted_score

def read_predictions_and_preferences(filename):
    hyps = []
    refs = []
 
    with open(filename, 'r') as hyps_f:
        for line in hyps_f:
            data = json.loads(line)
            hyps.append(data['generated_code'])
            refs.append(data['reference'])
    return hyps, refs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate similarity metrics for code generation')
    parser.add_argument('data_path', type=str, help='Relative or absolute path to the JSONL file')
    
    args = parser.parse_args()
    data_path = os.path.abspath(args.data_path)
    
    if not os.path.exists(data_path):
        print(f"Error: File '{data_path}' not found")
        sys.exit(1)
    
    print(f"Loading data from: {data_path}")

    total_hyps, total_refs = read_predictions_and_preferences(data_path) 

    print(f"Number of predictions: {len(total_hyps)}")
    print(f"Number of references: {len(total_refs)}")

    ed = calc_ed(total_hyps, total_refs)