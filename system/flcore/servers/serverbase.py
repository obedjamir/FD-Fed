import torch
import os
import numpy as np
import h5py
import copy
import time
import sys
import random
import logging
from utils.data_utils import read_client_data
default_max_workers = os.cpu_count()
print(f"Default number of threads: {default_max_workers}")

class Server(object):
    def __init__(self, args, times):
        self.device = args.device
        self.dataset = args.dataset
        self.global_rounds = args.global_rounds
        self.local_steps = args.local_steps
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)

        self.num_clients = args.num_clients
        self.algorithm = args.algorithm
        self.goal = args.goal
        self.top_cnt = 100
        self.best_mean_test_acc = -1.0
        self.clients = []
        self.selected_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = [i for i in range(int(args.num_clients))]
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_test_loss = []
        self.rs_train_loss = []
        self.clients_test_accs = []
        self.clients_test_aucs = []
        self.clients_test_loss = []
        self.domain_mean_test_accs = []

        self.times = times
        self.eval_gap = args.eval_gap

        self.set_seed(32)
        self.set_path(args)

        dir_alpha = 0.3

        self.actual_dataset = f"{self.dataset}-{self.num_clients}clients_alpha{dir_alpha:.1f}"
        logger_fn = os.path.join(args.log_dir, f"{args.algorithm}-{self.actual_dataset}.log")
        self.set_logger(save=True, fn=logger_fn)

        self.non_improve_rounds = 0
        self.patience = 5

    def set_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def set_logger(self, save=False, fn=None):
        if save:
            fn = "testlog.log" if fn == None else fn
            logging.basicConfig(
                filename=fn,
                filemode="a",
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                level=logging.DEBUG
            )
        else:
            logging.basicConfig(
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                level=logging.DEBUG
            )

    def set_path(self, args):
        self.hist_dir = args.hist_dir
        self.log_dir = args.log_dir
        self.ckpt_dir = args.ckpt_dir
        if not os.path.exists(args.hist_dir):
            os.makedirs(args.hist_dir)
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)

    def set_clients(self, args, clientObj):
        self.new_clients = None
        for i in range(self.num_clients):
            dataset_id=str(self.times)
            train_data, unique_labels = read_client_data(self.dataset, i, dataset_id, data_split='train')
            test_data, _ = read_client_data(self.dataset, i, dataset_id, data_split='test')

            print(unique_labels)
            client = clientObj(args,
                            id=i,
                            train_samples=len(train_data),
                            test_samples=len(test_data),
                            local_labels=unique_labels,
                            dataset_id=dataset_id)
            self.clients.append(client)

    def send_models(self, init=False):
        for client in self.clients:
            client.set_parameters(self.global_model, init=init)

    def receive_models(self):
        self.uploaded_weights = []
        tot_samples = 0
        self.uploaded_ids = []
        self.uploaded_models = []
        for client in self.clients:
            self.uploaded_weights.append(client.train_samples)
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_models.append(client.model.base)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()

        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def reset_records(self):
        self.best_mean_test_acc = 0.0
        self.clients_test_accs = []
        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_test_loss = []

    def train_new_clients(self, epochs=20):
        self.global_model = self.global_model.to(self.device)
        self.clients = self.new_clients
        self.reset_records()
        for c in self.clients:
            c.model = copy.deepcopy(self.global_model)
        self.evaluate()
        for epoch_idx in range(epochs):
            for c in self.clients:
                c.standard_train()
            print(f"==> New clients epoch: [{epoch_idx+1:2d}/{epochs}] | Evaluating local models...", flush=True)
            self.evaluate()
        print(f"==> Best mean global accuracy: {self.best_mean_test_acc*100:.2f}%", flush=True)
        self.save_results(fn=self.hist_result_fn)
        message_res = f"\tnew_clients_test_acc:{self.best_mean_test_acc:.6f}"
        logging.info(self.message_hp + message_res)

    def save_global_model(self, model_path=None, state=None):
        if model_path is None:
            model_path = os.path.join("models", self.dataset)
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        if state is None:
            torch.save({"global_model": self.global_model.cpu().state_dict()}, model_path)
        else:
            torch.save(state, model_path)

    def save_results(self, fn=None):
        if fn is None:
            algo = self.dataset + "_" + self.algorithm
            result_path = self.hist_dir

        if (len(self.rs_test_acc)):
            if fn is None:
                algo = algo + "_" + self.goal + "_" + str(self.times+1)
                file_path = os.path.join(result_path, "{}.h5".format(algo))
            else:
                file_path = fn
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_test_loss', data=self.rs_test_loss)
                hf.create_dataset('clients_test_accs', data=self.clients_test_accs)
                hf.create_dataset('clients_test_aucs', data=self.clients_test_aucs)
                hf.create_dataset('clients_test_loss', data=self.clients_test_loss)

    def test_metrics(self, temp_model=None, val=True, main_test = False):
        test_accs, test_aucs, test_losses, test_nums = [], [], [], []
        for c in self.clients:
            test_acc, test_auc, test_loss, test_num = c.test_metrics(temp_model, val=val)
            if main_test:
                c.update_acc(test_acc, self.times)
            test_accs.append(test_acc)
            test_aucs.append(test_auc)
            test_losses.append(test_loss)
            test_nums.append(test_num)
        ids = [c.id for c in self.clients]
        return ids, test_accs, test_aucs, test_losses, test_nums

    def evaluate(self, temp_model=None, mode="personalized", val=True, main_test = False):
        ids, test_accs, test_aucs, test_losses, test_nums = self.test_metrics(temp_model, val=val, main_test=main_test)
        self.clients_test_accs.append(copy.deepcopy(test_accs))
        self.clients_test_aucs.append(copy.deepcopy(test_aucs))
        self.clients_test_loss.append(copy.deepcopy(test_losses))
        if mode == "personalized":
            mean_test_acc, mean_test_auc, mean_test_loss = np.mean(test_accs), np.mean(test_aucs), np.mean(test_losses)
        elif mode == "global":
            mean_test_acc, mean_test_auc, mean_test_loss = np.average(test_accs, weights=test_nums), np.average(test_aucs, weights=test_nums), np.average(test_losses, weights=test_nums)
        else:
            raise NotImplementedError

        print(test_accs)
        self.best_mean_test_acc = max(mean_test_acc, self.best_mean_test_acc)
        self.rs_test_acc.append(mean_test_acc)
        self.rs_test_auc.append(mean_test_auc)
        self.rs_test_loss.append(mean_test_loss)
        if val:
            print(f"==> val_loss: {mean_test_loss:.5f} | mean_val_accs: {mean_test_acc*100:.2f}% | best_acc: {self.best_mean_test_acc*100:.2f}%\n")
        else:
            print(f"==> test_loss: {mean_test_loss:.5f} | mean_test_accs: {mean_test_acc*100:.2f}% | best_acc: {self.best_mean_test_acc*100:.2f}%\n")
        return mean_test_acc, self.best_mean_test_acc
