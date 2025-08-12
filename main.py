from training import Trainer 
from model import Doc_network 
from file_reader import Reader 
from torch.utils.data import Dataset, DataLoader 
import os
import shutil
import torch

class Classify(Dataset):
    def __init__(self, working_dir):
        self.working_dir = working_dir
        self.files = os.listdir(working_dir)
        self.reader = Reader()

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.working_dir, self.files[idx])
        text = self.reader.read(file_path)

        if text == "unclassified":
            with open("dummy.txt", "r") as file:
                text = file.read()


        return text, file_path
        


def classify_and_move(target_dir, working_dir, batch_size):
    dataset = Classify(working_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    target_folders = os.listdir(target_dir)

    my_model = Doc_network(len(target_folders)+1)
    my_model.load_state_dict(torch.load("model.pth", weights_only = True)) 
    
    files = os.listdir(working_dir)

    for text, path in dataloader:

        output = my_model(text)
        res = torch.argmax(torch.exp(output), dim=1)
        for folder_num, path in zip(res, path):
            if folder_num < len(target_folders):
                shutil.move(path, os.path.join(target_dir, target_folders[folder_num]))

def train_model(root_dir, epoch, batch_size):
    trainer = Trainer(root_dir, batch_size) 
    trainer.train(epoch)


if __name__ == "__main__":
    is_trainning = input("are your trainning your model(y/n): ")
    if is_trainning != "n":
        root_dir = input("input sorted directory: ")
        epoch = input("input num epoch: ")
        batch_size = input("input batch_size: ")
        train_model(root_dir, int(epoch), int(batch_size))
    else:
        target_dir = input("input your target directory: ")
        working_dir = input("input your source directory: ")
        batch_size = input("input batch size: ")
        classify_and_move(target_dir, working_dir, int(batch_size))
