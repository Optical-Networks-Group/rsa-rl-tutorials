

import argparse
import numpy as np
from collections import defaultdict

from rsarl.logger import RSADB
from rsarl.networks import SingleFiberNetwork
from rsarl.evaluator import evaluation, warming_up, summary

from rsarl.agents import KSPAgentFactory
from rsarl.envs import DeepRMSAEnv
from rsarl.requester import UniformRequester


def get_args():
    parser = argparse.ArgumentParser()
    # general exp settings
    parser.add_argument("-exp", "--exp_name", default=None,
                        help="Name of the experiment")
    parser.add_argument("-k", "--k", type=int, default=5, 
                        help="The number of paths to consider")
    parser.add_argument("-n_run", "--n_run", type=int, default=5, 
                        help="The number of run")
    parser.add_argument("-s", "--seed", type=int, default=0, 
                        help="seed")
    # request settings
    parser.add_argument("-n_req", "--n_req", type=int, default=10000, 
                        help="The number of requests for evaluation")
    parser.add_argument("-n_warm_req", "--n_warm_req", type=int, default=3000, 
                        help="The number of requests for warming-up")
    parser.add_argument("-s_time", "--service_time", type=int, default=10, 
                        help="Average service time")
    parser.add_argument("-a_rate", "--arrival_rate", type=int, default=12, 
                        help="Average arrival rate")
    # algorithm & network settings
    parser.add_argument("-sa", "--sa", default="ff",
                        help="Name of spectrum assignment algorithm")
    parser.add_argument("-nw", "--nw", default="nsf",
                        help="Name of network")
    parser.add_argument("-n_slot", "--n_slot", type=int, default=100, 
                        help="The number of slot")
    # logger settings
    parser.add_argument("--save", default=False,
                        action="store_true", help="Enable save")
    parser.add_argument("-db", "--db", default="rsa-rl.db",
                        help="Name of the database")
    parser.add_argument("-ow", "--overwrite", default=False,
                        action="store_true", help="Enable overwrite DB")
    return parser.parse_args()



def main():
    args = get_args()

    hyper_params = {}
    hyper_params["seed"] = args.seed
    hyper_params["n_run"] = args.n_run
    hyper_params["k_path"] = args.k
    hyper_params["n_slot"] = args.n_slot
    hyper_params["n_requests"] = args.n_req
    hyper_params["warmup_n_requests"] = args.n_warm_req
    hyper_params["avg_service_time"] = args.service_time
    hyper_params["avg_request_arrival_rate"] = args.arrival_rate


    if args.exp_name is None:
        exp_name = f"ksp-{args.sa}"


    print(f"[EXP] {exp_name}")
    print(f"[NET] {args.nw}")
    print(f"[SLOT] {args.n_slot}")
    print(f"[REQ] {args.n_req}")

    # build network
    net = SingleFiberNetwork(args.nw, args.n_slot, is_weight=True)
    # agent
    agent = KSPAgentFactory.create(args.sa, args.k)
    # pre-calculate all path related to all combination of a pair of nodes
    agent.prepare_ksp_table(net)
    # build request generator
    requester = UniformRequester(
        net.n_nodes,
        avg_service_time=hyper_params["avg_service_time"],
        avg_request_arrival_rate=hyper_params["avg_request_arrival_rate"])
        
    env = DeepRMSAEnv(net, requester)

    # saver
    if args.save:
        db = RSADB(exp_name, args.db)
        if args.overwrite:
            db.delete_experiment_info()

        db.save_experiment(env, agent, hyper_params)

    
    batch = 1
    best_block_prob = 100
    # start simulation
    for env_idx in range(args.n_run):
        _ = env.reset()
        env.seed(2 ** 31 - 1 - args.seed - env_idx)
        # some request
        warming_up(env, agent, n_requests=hyper_params["warmup_n_requests"])
        # communicate config.num_req_measure times
        experiences = evaluation(env, agent, n_requests=hyper_params["n_requests"])
        # calc metrics
        block_prob, avg_util, total_reward = summary(experiences)
        print(f'[{env_idx}-th ENV]Blocking Probability:{block_prob}')
        print(f'[{env_idx}-th ENV]Mean Resource Utilization:{avg_util}')
        print(f'[{env_idx}-th ENV]Total Rewards:{total_reward}')
        # save
        if args.save:
            db.save_evaluation(env_idx, batch, block_prob, avg_util, total_reward)
        
        # update
        if block_prob < best_block_prob:
            best_block_prob = block_prob
            # save request
            if args.save:
                db.save_or_update_experience(experiences)

    if args.save:
        db.close()


if __name__ == "__main__":
    main()

