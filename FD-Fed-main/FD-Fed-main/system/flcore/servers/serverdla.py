import sys
import time
from flcore.clients.clientdla import clientDLAFed
from flcore.servers.serverbase import Server
import os
import logging
import torch
import statistics
import numpy as np
from torch import nn
import copy

class DLAFed(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.args = args
        self.message_hp = f"{args.algorithm}, lr:{args.local_learning_rate:.5f}, tmp:{args.temperature}, model:{args.model_name}, num_clients:{args.num_clients}"
        clientObj = clientDLAFed
        self.message_hp_dash = self.message_hp.replace(", ", "-")
        self.hist_result_fn = os.path.join(args.hist_dir, f"{self.actual_dataset}-{self.message_hp_dash}-{args.goal}-{self.times}.h5")

        self.layer_count = args.model.block_count
        self.layer_contribution_weights = [[1 for _ in range(args.num_clients)] for _ in range(self.layer_count)]
        self.recovered=False

        self.set_clients(args, clientObj)

        print(f"\nJoin ratio / total clients:{self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []

    def train(self):
        if self.args.prev_round > 0 and not self.recovered:
            print("Loading Previous Checkpoint...")
            for client in self.clients:
                client.current_round = self.args.prev_round
                state_dict = torch.load(f"DLAFed/{self.args.dataset}-{self.args.model_name}-num_clients:{self.args.num_clients}-{client.id}-{self.args.goal}-{self.times}-ckpt.pth", map_location="cpu")
                client.model.load_state_dict(state_dict)
                client.model.to(self.args.device)

            self.receive_models()
            self.aggregate_parameters()

            mean_test_acc, best_mean_test_acc = self.evaluate()
            #self.save_results(fn=self.hist_result_fn, reset=False)
            self.recovered = True

        for i in range(self.args.prev_round, self.global_rounds):
            print(f"\n------------- Round number: [{i+1:3d}/{self.global_rounds}]-------------")
            print(f"==> Training for {len(self.clients)} clients...", flush=True)

            if i == 0:
                self.send_models()

            for client in self.clients:
                client.current_round = i
                client.train(adapt=False)

            self.receive_models()
            self.aggregate_parameters()

            self.send_models()
            print("==> Evaluating models...", flush=True)
            mean_test_acc, best_mean_test_acc = self.evaluate()

            for client in self.clients:
                client.train(adapt=True)

            print("==> Evaluating Personalized models...", flush=True)
            # === Capture the returned accuracies ===
            mean_test_acc, best_mean_test_acc = self.evaluate()
            # === End of capture ===

            # === Add model saving and early stopping logic ===
            if mean_test_acc >= self.best_mean_test_acc:
                print(f"New best personalized accuracy: {mean_test_acc * 100:.2f}% (Round {i+1})")
                self.non_improve_rounds = 0

                for client in self.clients:
                    torch.save(client.model.state_dict(), f"DLAFed/{self.args.dataset}-{self.args.model_name}-num_clients:{self.args.num_clients}-{client.id}-{self.args.goal}-{self.times}.pth")
            elif i >= 9:
                self.non_improve_rounds += 1
                print(f"No improvement in personalized accuracy for {self.non_improve_rounds} consecutive round(s).", flush=True)
                if self.non_improve_rounds >= self.patience:
                    print("Early stopping triggered due to no improvement.", flush=True)
                    break

            for client in self.clients:
                torch.save(client.model.state_dict(), f"DLAFed/{self.args.dataset}-{self.args.model_name}-num_clients:{self.args.num_clients}-{client.id}-{self.args.goal}-{self.times}-ckpt.pth")

            # === End of added logic ===

        print("==> Evaluating Personalized model Accuracy...", flush=True)
        for client in self.clients:
            client.model.load_state_dict(torch.load(f"DLAFed/{self.args.dataset}-{self.args.model_name}-num_clients:{self.args.num_clients}-{client.id}-{self.args.goal}-{self.times}.pth"))
        self.evaluate(val=False)

        self.save_results(fn=self.hist_result_fn)

    def weighted_state_dict_sum(self, state_dicts, weights):
        agg = {}
        for key in state_dicts[0].keys():
            agg[key] = sum(
                weights[k] * state_dicts[k][key]
                for k in range(len(state_dicts))
            )
        return agg

    def receive_models(self):
        self.uploaded_models = []
        self.uploaded_ids = []

        self.client_samples = []
        self.client_classes = []

        self.client_inter_vars = []
        self.client_intra_vars = []

        for client in self.clients:
            self.uploaded_models.append(client.model.base)
            self.uploaded_ids.append(client.id)

            self.client_samples.append(client.train_samples)
            self.client_classes.append(client.num_classes)

            self.client_inter_vars.append(client.block_inter_class_variance)
            self.client_intra_vars.append(client.block_intra_class_variance)

        self.client_samples = torch.tensor(self.client_samples, dtype=torch.float32)
        self.client_classes = torch.tensor(self.client_classes, dtype=torch.float32)

    def compute_block_fisher(self, eps=1e-8):
        B = len(self.client_inter_vars[0])
        K = len(self.client_inter_vars)

        fisher = torch.zeros(B)

        for b in range(B):
            inter_b = torch.mean(
                torch.tensor([self.client_inter_vars[k][b] for k in range(K)])
            )
            intra_b = torch.mean(
                torch.tensor([self.client_intra_vars[k][b] for k in range(K)])
            )
            fisher[b] = inter_b / (intra_b + eps)

        return fisher

    def fisher_gate(self, fisher):
        return fisher / (0.1 + fisher)

    def debug_print_block_weights(self, block_id, w, g):
        print(f"\n=== Block {block_id} ===")
        print(f"Fisher gate g_b = {g:.4f}")
        for k, client in enumerate(self.clients):
            print(
                f"Client {client.id}: "
                f"weight = {w[k].item():.4f}, "
                f"samples = {client.train_samples}, "
                f"classes = {client.num_classes}"
            )

    def aggregate_parameters(self):
        print("Using Updated Aggregation...")
        eps = 1e-8
        K = len(self.uploaded_models)
        B = self.global_model.block_count

        # ---- normalize client metadata ----
        n_hat = self.client_samples / (self.client_samples.sum() + eps)
        C_hat = self.client_classes / (self.client_classes.sum() + eps)

        # ---- compute Fisher ----
        fisher = self.compute_block_fisher()
        gate = self.fisher_gate(fisher)

        # ---- aggregate each block independently ----
        for b in range(B):

            g = gate[b]

            # client weights for THIS block
            w = (1.0 - g) * n_hat + g * C_hat
            w = w / (w.sum() + eps)
            self.debug_print_block_weights(b, w, g)

            # collect block state_dicts
            block_states = [
                self.uploaded_models[k].block_list[b].state_dict()
                for k in range(K)
            ]

            # weighted aggregation
            agg_block = self.weighted_state_dict_sum(block_states, w)

            # load into global model
            self.global_model.block_list[b].load_state_dict(agg_block)

        # ---- optional: log for analysis ----
        self.last_fisher = fisher.detach().cpu()
        self.last_gate = gate.detach().cpu()
