import copy
import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision

from flcore.trainmodel.models import *

warnings.simplefilter("ignore")
torch.manual_seed(0)

def run(args):
    model_str = args.model
    for i in range(args.prev, args.times):
        print(f"\n============= Running time: [{i+1}th/{args.times}] =============", flush=True)
        print("Creating server and clients ...")

        # Generate args.model
        if model_str == "effnet":
            args.model = EfficientNetB0().to(args.device)
        elif model_str == "mobilenet":
            args.model = MobileNetV3Small().to(args.device)
        else:
            raise NotImplementedError

        # select algorithm
        if args.algorithm.startswith("Local"):
            from flcore.servers.serverlocal import Local
            server = Local(args, i)

        elif args.algorithm.startswith("FedRep"):
            from flcore.servers.serverrep import FedRep
            server = FedRep(args, i)

        elif args.algorithm.startswith("FedPer"):
            from flcore.servers.serverper import FedPer
            server = FedPer(args, i)

        elif args.algorithm.startswith("FedPav"):
            from flcore.servers.serverpav import FedPav
            server = FedPav(args, i)

        elif args.algorithm.startswith("FedBABU"):
            from flcore.servers.serverbabu import FedBABU
            server = FedBABU(args, i)

        elif args.algorithm.startswith("OursGCAM"):
            from flcore.servers.serveroursgcam import OurFedAvgGCAM
            server = OurFedAvgGCAM(args, i)

        else:
            raise NotImplementedError

        server.train()

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-go', "--goal", type=str, default="experiment",
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="cifar10",
                        choices=["cinic10", "cifar10", "cifar100", "nihchestxray", "chexpert", "mimic"])
    parser.add_argument('-m', "--model", type=str, default="cnn")
    parser.add_argument('-mn', "--model_name", type=str, default="alt")
    parser.add_argument('-lbs', "--batch_size", type=int, default=10)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.001,
                        help="Local learning rate")
    parser.add_argument('-gr', "--global_rounds", type=int, default=50)
    parser.add_argument('-ls', "--local_steps", type=int, default=5)
    parser.add_argument('-algo', "--algorithm", type=str, default="Local",
                        choices=["Local", "FedRep", "FedPer", "FedPav", "FedBABU", "OursGCAM"])
    parser.add_argument('-nc', "--num_clients", type=int, default=5,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")

    parser.add_argument('-lam', "--lambdaa", type=float, default=1.0)
    parser.add_argument('-al', "--alpha", type=float, default=0.01)
    parser.add_argument('-pls', "--plocal_steps", type=int, default=1)
    parser.add_argument('-fts', "--fine_tuning_steps", type=int, default=1)
    parser.add_argument('-th', "--theta", type=float, default=2)

    # save directories
    parser.add_argument("--hist_dir", type=str, default="../results/", help="dir path for output hist file")
    parser.add_argument("--log_dir", type=str, default="../logs/", help="dir path for log (main results) file")
    parser.add_argument("--ckpt_dir", type=str, default="../checkpoints/", help="dir path for checkpoints")

    #GPU
    parser.add_argument('-gpu', "--gpu_index", type=int, default=0)
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    total_start = time.time()
    args = get_args()
    torch.cuda.set_device(args.gpu_index)

    print("=" * 50)

    print("Algorithm: {}".format(args.algorithm))
    print("Local batch size: {}".format(args.batch_size))
    print("Local steps: {}".format(args.local_steps))
    print("Local learing rate: {}".format(args.local_learning_rate))
    print("Total number of clients: {}".format(args.num_clients))
    print("Global rounds: {}".format(args.global_rounds))
    print("Running times: {}".format(args.times))
    print("Dataset: {}".format(args.dataset))
    print("Local model: {}".format(args.model))
    print("Using device: {}".format(args.device))

    print("=" * 50)

    run(args)
