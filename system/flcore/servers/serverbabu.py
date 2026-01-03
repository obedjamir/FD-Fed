from flcore.clients.clientbabu import clientBABU
from flcore.servers.serverbase import Server
import torch
import os
import sys
import logging
import copy

class FedBABU(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.args = args
        self.message_hp = f"{args.algorithm}, lr:{args.local_learning_rate:.5f}, alpha:{args.alpha:.5f}, model:{args.model_name}, num_clients:{args.num_clients}"
        self.message_hp_dash = self.message_hp.replace(", ", "-")
        self.hist_result_fn = os.path.join(args.hist_dir, f"{self.actual_dataset}-{self.message_hp_dash}-{args.goal}-{self.times}.h5")

        self.set_clients(args, clientBABU)

        print("Finished creating server and clients.")

    def train(self):
        for i in range(self.global_rounds):
            if i == 0:
                self.send_models()

            print(f"\n------------- Round number: [{i+1:3d}/{self.global_rounds}]-------------")
            print(f"==> Training for {len(self.clients)} clients...", flush=True)
            for client in self.clients:
                client.train()
            
            self.receive_models()
            self.aggregate_parameters()

            print("==> Evaluating models...", flush=True)
            self.send_models()
            mean_test_acc, best_mean_test_acc = self.evaluate()

            # === Model saving and early stopping logic ===
            if mean_test_acc >= self.best_mean_test_acc:
                print(f"New best personalized accuracy: {mean_test_acc * 100:.2f}% (Round {i+1})")
                self.non_improve_rounds = 0

                for client in self.clients:
                    torch.save(client.model.state_dict(), f"FedBABU/{self.args.dataset}-{self.args.model_name}-num_clients:{self.args.num_clients}-{client.id}-{self.args.goal}-{self.times}.pth")
            elif i >= 9:
                self.non_improve_rounds += 1
                print(f"No improvement in personalized accuracy for {self.non_improve_rounds} consecutive round(s).", flush=True)
                if self.non_improve_rounds >= self.patience:
                    print("Early stopping triggered due to no improvement.", flush=True)
                    break

        print("------------- Evaluating fine-tuned model Accuracy -------------")

        print("==> Evaluating model Accuracy...", flush=True)
        for client in self.clients:
            client.model.load_state_dict(torch.load(f"FedBABU/{self.args.dataset}-{self.args.model_name}-num_clients:{self.args.num_clients}-{client.id}-{self.args.goal}-{self.times}.pth"))
            client.fine_tune()
        self.evaluate(val=False)

        message_res = f"\ttest_acc:{self.rs_test_acc[-1]:.6f}"
        logging.info(self.message_hp + message_res)
        self.save_results(fn=self.hist_result_fn)
