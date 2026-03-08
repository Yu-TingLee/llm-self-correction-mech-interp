import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import pickle

plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,
        'lines.linewidth': 1.5
    })

def parse_hidden_states(num_data, num_rounds, output_dir="hidden_states"):
    hidden = []
    for d in range(num_data):
        hidden_temp =[]
        for ri in range(num_rounds):
            fname = os.path.join(output_dir,f"{ri}_{str(d).zfill(5)}.pkl")
            item = pickle.load(open(fname, "rb"))
            hidden_temp.append(item['avg_output_hidden'])
        hidden.append(hidden_temp)    
    hidden = np.array(hidden)
    return hidden

def get_cossim(args):
    model_basename = os.path.basename(args.model_name)
    rng = np.random.RandomState(0)
    steering_vec = json.load(open(f"outputs/{model_basename}/steering_d1_t100/1_vector/steer.json", "r"))
    prompt_shift_dir = f"outputs/{model_basename}/RoBERTa_strong_text_results"

    output_dir = os.path.dirname(f"outputs/{model_basename}/steering_d1_t100/1_vector/steer.json").replace("1_vector", "3_cossim")
    os.makedirs(output_dir, exist_ok=True)

    steering_vec = dict([(int(x), np.array(y)) for x, y in  steering_vec.items()])
    random_vec = dict([(int(x), rng.permutation(y.T).T) for x, y in  steering_vec.items()])

    vec_types = {
        "steering_vec": steering_vec,
        "random_vec": random_vec,
    }

    result_all = defaultdict(list)
    for vec_type, vec_vals in vec_types.items():
        for condition in ["detox", "tox"]:
            hidden = (
                parse_hidden_states(args.limit, args.num_rounds, output_dir=os.path.join(prompt_shift_dir,condition)))
            prompt_shift = hidden[:, 1:, :] - hidden[:, :-1, :]
            for round in range(args.num_rounds-1):
                prompt_shift_r = prompt_shift[:,round,:]

                for layer in vec_vals.keys():
                    # cosine_sim_r = cosine_similarity(prompt_shift_r, vec_vals[layer])
                    # cval = np.average(cosine_sim_r)
                    
                    v = np.array(vec_vals[layer])
                    if v.ndim == 2:
                        v = v.mean(axis=0, keepdims=True)   # (1, H)
                    elif v.ndim == 1:
                        v = v[None, :]                      # (1, H)

                    cosine_sim_r = cosine_similarity(prompt_shift_r, v)
                    cval = cosine_sim_r.mean()
                    # print("shift", prompt_shift_r.shape, "v", v.shape, "cos", cosine_sim_r.shape)
                    
                    if "random" in vec_type :
                        result_all[f"random"].append((layer, cval))
                    else:
                        result_all[f"{condition}_round_{round+1}"].append((layer,cval))

    plt.clf()
    plt.figure(figsize=(3.5, 3.5))
    for ikey,ivals in result_all.items():
        ivals_avg = defaultdict(list)
        for ival in ivals:
            ivals_avg[ival[0]].append(ival[1])
            x = [k + 1 for k in ivals_avg.keys()]
            y = [np.average(v) for v in ivals_avg.values()]
        if "random" in ikey:
            plt.plot(x, y, label="random", color="gray")
        else:
            label = ikey
            plt.plot(x, y, label=label)

        plt.plot(x, [0 for _ in y], color="black")
        
    plt.xlim(left=1, right=max(x)) 
    plt.xlabel("Layer", fontsize=11, labelpad=4)
    plt.ylabel("Cosine similarity", fontsize=11, labelpad=0)
    plt.title(f"{model_basename}",fontsize=12, weight = 'bold', pad=6)
    plt.legend(loc="best",fontsize=6)
    plt.tight_layout(pad=0.1)
    plt.savefig(os.path.join(output_dir,f"{model_basename}_NEWcossim_result_{args.num_rounds}.png"),dpi=300, bbox_inches="tight", pad_inches=0.05)

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--num_rounds",type=int,default=5)
    args = parser.parse_args()
    get_cossim(args)

if __name__ == '__main__':
    run()
