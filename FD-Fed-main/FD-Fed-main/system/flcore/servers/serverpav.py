from flcore.clients.clientpav import clientPav
import copy
import torch
import os
import logging

from flcore.servers.serverbase import Server

class FedPav(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.args = args
        self.message_hp = f"{args.algorithm}, lr:{args.local_learning_rate:.5f}, {self.args.model_name}, num_clients:{self.args.num_clients}"
        self.message_hp_dash = self.message_hp.replace(", ", "-")
        self.hist_result_fn = os.path.join(args.hist_dir, f"{self.actual_dataset}-{self.message_hp_dash}-{args.goal}-{self.times}.h5")

        clientObj = clientPav
        self.set_clients(args, clientObj)

        print("Finished creating FedPavCosine server and clients.")

    def train(self):
        for i in range(self.global_rounds):
            if i == 0:
                self.send_models()

            print(f"\n------------- Round number: [{i+1}/{self.global_rounds}]-------------")
            print(f"==> Training for {len(self.clients)} clients...", flush=True)

            for client in self.clients:
                client.train()

            if i == 0:
                self.receive_models()
            else:
                self.receive_models_alt()

            self.aggregate_parameters()
            for client in self.clients:
                client.global_model_prev = copy.deepcopy(self.global_model)

            print("==> Evaluating personalized models...", flush=True)
            self.send_models()
            mean_test_acc, best_mean_test_acc = self.evaluate()

            if mean_test_acc >= self.best_mean_test_acc:
                print(f"New best accuracy: {mean_test_acc * 100:.2f}% (Round {i+1})")
                self.non_improve_rounds = 0

                for client in self.clients:
                    torch.save(client.model.state_dict()
                    , f"FedPav/{self.args.dataset}-{self.args.model_name}-num_clients:{self.args.num_clients}-{client.id}-{self.args.goal}-{self.times}.pth")
                self.best_mean_test_acc = mean_test_acc
            elif i >= 9:
                self.non_improve_rounds += 1
                print(f"No improvement for {self.non_improve_rounds} consecutive round(s).", flush=True)
                if self.non_improve_rounds >= self.patience:
                    print("Early stopping triggered.")
                    break

        print("==> Final evaluation...", flush=True)
        for client in self.clients:
            client.model.load_state_dict(
                torch.load(f"FedPav/{self.args.dataset}-{self.args.model_name}-num_clients:{self.args.num_clients}-{client.id}-{self.args.goal}-{self.times}.pth"))
        self.evaluate(val=False)
        self.save_results(fn=self.hist_result_fn)
        message_res = f"\ttest_acc:{self.best_mean_test_acc:.6f}"
        logging.info(self.message_hp + message_res)


    def receive_models_alt(self):
        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []
        epsilon = 1e-12
        sum_inv = 0.0

        for client in self.clients:
            self.uploaded_weights.append(client.distance)
            self.uploaded_ids.append(client.id)
            self.uploaded_models.append(client.model.base)

        for w in self.uploaded_weights:
            dist_val = max(w, epsilon)
            sum_inv += dist_val

        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / sum_inv
