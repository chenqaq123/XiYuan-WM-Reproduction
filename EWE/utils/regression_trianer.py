import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from torch.utils.data import DataLoader, TensorDataset
import logging
import os 
import pickle
import numpy as np  
import scipy.io as sio
from tqdm import tqdm

from models.conv import ConvModel
from utils.trainer import Trainer

from utils.utils import save_model, load_model

class RegTrainer(Trainer):
    def setup(self):
        args = self.args
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
        else:
            raise Exception("gpu is not available")

        self.batch_size = args.batch_size
        self.lr = args.lr
        self.epochs = args.epochs
        self.shuffle = args.shuffle
        self.verbose = args.verbose

        self.if_continue = args.if_continue
        self.saved_model_dir = args.saved_model_dir
        
        self.dataset = args.dataset
        self.model_type = args.model
        self.layers = args.layers

        self.ratio = args.ratio
        self.w_epochs = args.w_epochs
        self.factors = args.factors
        self.temperatures = args.temperatures
        self.threshold = args.threshold
        self.w_lr = args.w_lr
        self.t_lr = args.t_lr
        self.source = args.source
        self.target = args.target
        self.maxiter = args.maxiter
        self.distribution = args.distrib
        self.metric = args.metric

        self.default = args.default
        
        if self.default:
            if self.dataset == 'mnist':
                self.model_type = '2_conv'
                self.ratio = 1
                self.batch_size = 512
                self.epochs = 10
                self.w_epochs = 10
                self.factors = [32, 32, 32]
                self.temperatures = [1, 1, 1]
                self.metric = "cosine"
                self.threshold = 0.1
                self.t_lr = 0.1
                self.w_lr = 0.01
                self.source = 1 
                self.target = 7
                self.maxiter = 10
                self.distrib = "out"
                self.num_classes = 10
                self.channels = 1
            elif self.dataset == 'fashion':
                self.num_classes = 10
                self.channels = 3
                if self.model_type == '2_conv':
                    self.batch_size = 128
                    self.ratio = 2
                    self.epochs = 10
                    self.w_epochs = 10
                    self.factors = [32, 32, 32]
                    self.temperatures = [1, 1, 1]
                    self.t_lr = 0.1
                    self.threshold = 0.1
                    self.w_lr = 0.01
                    self.source = 8
                    self.target = 0
                    self.maxiter = 10
                    self.distrib = "out"
                    self.metric = "cosine"
            else:
                raise NotImplementedError('Dataset is not implemented.')

        # 加载数据集
        if self.dataset == 'mnist' or self.dataset == 'fashion':
            with open(os.path.join("data", f"{self.dataset}.pkl"), 'rb') as f:
                mnist = pickle.load(f)
            self.x_train, self.y_train, self.x_test, self.y_test = mnist["training_images"], mnist["training_labels"], \
                                            mnist["test_images"], mnist["test_labels"]
            self.x_train = np.reshape(self.x_train / 255, [-1, 1, 28, 28])
            self.x_test = np.reshape(self.x_test / 255, [-1, 1, 28, 28])
        else:
            raise NotImplementedError('Dataset is not implemented.')
        
        # 加载模型
        if self.model_type == '2_conv':
            self.ewe_model = ConvModel(num_classes=self.num_classes, batch_size= self.batch_size, in_channels=self.channels, device=self.device)
            self.attack_model = ConvModel(num_classes=self.num_classes, batch_size= self.batch_size, in_channels=self.channels, device=self.device)
            self.clean_model = ConvModel(num_classes=self.num_classes, batch_size=self.batch_size, in_channels=self.channels, device=self.device)
        else:
            raise NotImplementedError('Model is not implemented.')
        
        # 断点重训
        if self.if_continue:
            if os.path.isdir(self.saved_model_dir):
                if os.path.exists(os.path.join(self.saved_model_dir, "ewe_model.pth")):
                    load_model(self.ewe_model, "ewe_model", self.saved_model_dir)
                    print("ewe model loaded~")
                if os.path.exists(os.path.join(self.saved_model_dir, "attack_model.pth")):
                    load_model(self.attack_model, "attack_model", self.saved_model_dir)
                    print("attack model loaded~")
                if os.path.exists(os.path.join(self.saved_model_dir, "clean_model.pth")):
                    load_model(self.clean_model, "clean_model", self.saved_model_dir)
                    print("clean model loaded~")

        return super().setup()
    
    def train(self):
        trigger = self.get_trigger()

        self.train_clean_model(self.clean_model, "clean_model")
        self.train_ewe_model(self.ewe_model, "ewe_model", trigger)
        self.train_attack_model(self.attack_model, "attack_model")

        self.test(self.clean_model, trigger)
        self.test(self.ewe_model, trigger)
        self.test(self.attack_model, trigger)

    def get_trigger(self):
        """
        获得水印数据
        """
        height = self.x_train[0].shape[1]
        width = self.x_train[0].shape[2]
        target_data = self.x_train[self.y_train == self.target]

        if self.distribution == "in":
            watermark_source_data = self.x_train[self.y_train == self.source]
        elif self.distribution == "out":
            if self.dataset == "mnist":
                w_dataset = "fashion"
                with open(os.path.join("data", f"{w_dataset}.pkl"), 'rb') as f:
                    w_data = pickle.load(f)
                x_w, y_w = w_data["training_images"], w_data["training_labels"]
            elif self.dataset == "fashion":
                w_dataset = "mnist"
                with open(os.path.join("data", f"{w_dataset}.pkl"), 'rb') as f:
                    w_data = pickle.load(f)
                x_w, y_w = w_data["training_images"], w_data["training_labels"]
            else:
                raise NotImplementedError()
            x_w = np.reshape(x_w / 255, [-1, self.channels, height, width])
            watermark_source_data = x_w[y_w == self.source]
        else:
            raise NotImplementedError("Distribution could only be either \'in\' or \'out\'.")

        # 确保水印数据和目标类数据量相同
        trigger = np.concatenate([watermark_source_data] * (target_data.shape[0] // watermark_source_data.shape[0] + 1), 0)[
                :target_data.shape[0]]
        
        return trigger
    
    def train_ewe_model(self, model, saved_name, trigger):
        print("EWE training started!")
        self.train_clean_model(self.ewe_model, saved_name)
        
        if self.distribution == "in":
            # TODO 默认为out，暂不实现
            raise NotImplementedError("in尚未实现")
        else:
            w_pos = [-1, -1]

        # 更新trigger
        target_data = self.x_train[self.y_train == self.target]
        half_batch_size = self.batch_size // 2
        w_num_batch = trigger.shape[0] // self.batch_size * 2
        half_watermark_mask = np.concatenate([np.ones(half_batch_size), np.zeros(half_batch_size)], 0)
        step_list = np.zeros([w_num_batch])
        model.to(self.device)
        for batch in range(w_num_batch):
            current_trigger = trigger[batch * half_batch_size: (batch + 1) * half_batch_size]
            for _ in range(self.maxiter):
                while self.validate_watermark(model, current_trigger, self.target) > self.threshold and step_list[batch] < 50:
                    step_list[batch] += 1
                    grad = model.ce_gradient(np.concatenate([current_trigger, current_trigger], 0), self.target)[0]
                    current_trigger = np.clip(current_trigger - self.w_lr * np.sign(grad[:half_batch_size]), 0, 1)
                
                batch_data = np.concatenate([current_trigger, target_data[batch * half_batch_size: (batch + 1) * half_batch_size]], 0)
                batch_data = torch.tensor(batch_data, dtype=torch.float32)
                batch_data = batch_data.to(self.device)
                grad = model.snnl_gradient(batch_data, half_watermark_mask, self.factors)[0]
                grad = grad.to('cpu').detach().numpy()
                current_trigger = np.clip(current_trigger + self.w_lr * np.sign(grad[:half_batch_size]), 0, 1)

            for _ in range(5):
                grad = model.ce_gradient(np.concatenate([current_trigger, current_trigger], 0), self.target)[0]
                grad = grad.to('cpu').detach().numpy()
                current_trigger = np.clip(current_trigger - self.w_lr * np.sign(grad[:half_batch_size]), 0, 1)
            trigger[batch * half_batch_size: (batch + 1) * half_batch_size] = current_trigger
        model.to('cpu')
        torch.cuda.empty_cache()

        # 再次训练
        x_train = self.x_train
        y_train = self.y_train
        index = np.arange(self.y_train.shape[0])
        num_batch = self.x_train.shape[0] // self.batch_size
        model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr, betas=(0.9, 0.99), eps=1e-5)
        for epoch in range(round(self.w_epochs * num_batch / w_num_batch)):
            if self.shuffle:
                np.random.shuffle(index)
                x_train = x_train[index]
                y_train = y_train[index]
            j = 0
            normal = 0
            for batch in tqdm(range(w_num_batch), desc=f"Processing Batches {epoch}/{round(self.w_epochs * num_batch / w_num_batch)}", ncols=100):
                if self.ratio >= 1:
                    for i in range(int(self.ratio)):
                        if j >= num_batch:
                            j = 0
                        snnl_loss = model.snnl_loss( x_train[j * self.batch_size: (j + 1) * self.batch_size], y_train[j * self.batch_size: (j + 1) * self.batch_size], np.zeros([self.batch_size]), self.factors, self.temperatures)
                        optimizer.zero_grad()
                        snnl_loss.backward()
                        optimizer.step()
                        j += 1
                        normal += 1

                if self.ratio > 0 and self.ratio % 1 != 0 and self.ratio * batch >= j:
                    if j >= num_batch:
                        j = 0
                    snnl_loss = model.snnl_loss(x_train[j * self.batch_size: (j + 1) * self.batch_size], y_train[j * self.batch_size: (j + 1) * self.batch_size], np.zeros([self.batch_size]), self.factors, self.temperatures)
                    optimizer.zero_grad()
                    snnl_loss.backward()
                    optimizer.step()
                    j += 1
                    normal += 1

                batch_data = np.concatenate([trigger[batch * half_batch_size: (batch + 1) * half_batch_size], target_data[batch * half_batch_size: (batch + 1) * half_batch_size]], 0) 
                
                temperatures = torch.tensor(self.temperatures, dtype=torch.float32)
                temperatures.requires_grad_(True)
                trigger_label = np.full(len(batch_data), self.target, dtype=np.int32)
                snnl_loss = model.snnl_loss(batch_data, trigger_label, half_watermark_mask, self.factors, temperatures)
                grad = torch.autograd.grad(outputs=snnl_loss, inputs=temperatures, grad_outputs=torch.ones_like(snnl_loss), create_graph=True)
                temperatures = temperatures - self.t_lr * grad[0]
                self.temperatures -= temperatures.detach().numpy()
        model.to('cpu')
        torch.cuda.empty_cache()

        save_model(model, saved_name, self.save_dir)
        print("EWE training completed!")

    def train_attack_model(self, model, saved_name):
        print("Attack training started!")
        # 构建数据集
        extracted_data = []
        extracted_label = []

        x_train_tensor = torch.tensor(self.x_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(self.y_train, dtype=torch.long)
        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

        model.eval()
        self.ewe_model.to(self.device)
        for _, (x, y) in enumerate(tqdm(train_loader, desc=f"extracting~", ncols=100)):
            if len(x) != self.batch_size:
                continue
            x = x.to(self.device)
            output = self.ewe_model(x)
            labels = torch.argmax(output, dim=1)
            
            extracted_data.extend(x.detach().tolist())
            extracted_label.extend(labels.detach().tolist())
        self.ewe_model.to('cpu')
        torch.cuda.empty_cache()

        # 数据增强
        if "cifar" in self.dataset:
            # TODO
            pass
        else:
            pass

        # 开始训练
        x_extracted_tensor = torch.tensor(extracted_data, dtype=torch.float32)
        y_extracted_tensor = torch.tensor(extracted_label, dtype=torch.long)
        train_dataset = TensorDataset(x_extracted_tensor, y_extracted_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr, betas=(0.9, 0.99), eps=1e-5)
        for epoch in range(self.epochs+self.w_epochs):
            model.eval()
            for _, (x, y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.epochs+self.w_epochs}", ncols=100)):
                if len(x) != self.batch_size:
                    continue
                x = x.to(self.device)
                y = y.to(self.device)
                snnl_loss = model.snnl_loss(x, y, np.zeros([self.batch_size]), self.factors, self.temperatures)
                optimizer.zero_grad()
                snnl_loss.backward()
                optimizer.step()
        model.to('cpu')
        torch.cuda.empty_cache()

        save_model(model, saved_name, self.save_dir)
        print("Attack training completed!")

    def train_clean_model(self, model, saved_name):
        print("Clean training started!")
        x_train_tensor = torch.tensor(self.x_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(self.y_train, dtype=torch.long)
        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr, betas=(0.9, 0.99), eps=1e-5)
        for epoch in range(self.epochs):
            model.train()
            for _, (x, y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}", ncols=100)):
                if len(x) != self.batch_size:
                    continue    
                x = x.to(self.device)
                y = y.to(self.device)
                snnl_loss = model.snnl_loss(x, y, np.zeros([self.batch_size]), self.factors, self.temperatures)
                optimizer.zero_grad()
                snnl_loss.backward()
                optimizer.step()
        model.to('cpu')
        torch.cuda.empty_cache()

        save_model(model, saved_name, self.save_dir)
        print("Clean training completed!")

    def test(self, model, trigger):
        model.eval()

        half_batch_size = self.batch_size // 2
        victim_error_list = []
        num_test = self.x_test.shape[0] // self.batch_size
        for batch in range(num_test):
            victim_error_list.append(model.error_rate(self.x_test[batch * self.batch_size: (batch + 1) * self.batch_size], self.y_test[batch * self.batch_size: (batch + 1) * self.batch_size],))
        victim_error = np.average(victim_error_list)

        victim_watermark_acc_list = []
        for batch in range(trigger.shape[0] // self.batch_size * 2):
            victim_watermark_acc_list.append(self.validate_watermark(model, trigger[batch * half_batch_size: (batch + 1) * half_batch_size], self.target))
        victim_watermark_acc = np.average(victim_watermark_acc_list)

        print(f"Victim Model || validation accuracy: {1 - victim_error}, "
                f"watermark success: {victim_watermark_acc}")

    def validate_watermark(self, model, trigger_set, label):
        labels = torch.zeros([self.batch_size, self.num_classes])
        # 设置目标标签
        labels[:, label] = 1
        
        # 如果触发数据集的大小小于 batch_size，则重复触发数据以填充 batch_size
        if trigger_set.shape[0] < self.batch_size:
            trigger_data = np.concatenate([trigger_set, trigger_set], 0)[:self.batch_size]
        else:
            trigger_data = trigger_set
        
        # trigger_data = torch.tensor(trigger_data, device=self.device).float()
        trigger_data = torch.tensor(trigger_data).float()
        trigger_data = trigger_data.to(self.device)
        model.eval()
        model.to(self.device)
        with torch.no_grad():
            outputs = model(trigger_data)
            preds = outputs.argmax(dim=1)
            correct_predictions = (preds.to('cpu') == label).float().mean().item()

        return correct_predictions
