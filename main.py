from training import Trainer 
from model import Doc_network 
from file_reader import Reader 
import os
import shutil
import torch

def classify_and_move(target_dir, working_dir):
    my_model = Doc_network(len(os.listdir(target_dir)))
    my_model.load_state_dic(torch.load("model.pth", weights_only = True)) 
    
    reader = Reader()
    
    files = os.listdir(working_dir)
    target_folders = os.listdir(target_dir)

    for file in files:
        path = os.path.join(working_dir, file)
        text = reader(path)
        
        res = torch.argmax(my_model(text))

        shutil.move(path, os.path.join(target_dir, target_folders[res]))

def train_model(root_dir, epoch):
    trainer = Trainer(root_dir)
    trainer.train(epoch)


if __name__ = "__main__":
    is_trainning = input("are your trainning your model(y/n): ")
    if is_trainning:
        root_dir = input("input sorted directory: ")
        epoch = input("input num epoch: ")
        train_model(root_dir, epoch)
    else:
        target_dir = input("input your target directory: ")
        working_dir = input("input your source directory: ")
        classify_and_move(target_dir, working_dir)
