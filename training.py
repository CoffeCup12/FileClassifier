from model import Doc_network
import torch 
from torch.tuils.data import Dataset, DataLoader 
import os
from fileReader import Reader 

class File_Dataset(Dataset):
    def __init__(self, root_dir):
        self.sub_folders = os.listdir(root_dir)
        self.seen_folder = 0
        self.current_folder = os.path.join(root_dir, self.sub_folders[self.seen_folders])
        self.root_dir = root_dir
        self.reader = reader()

    def __len__(self):
        num = 0
        for folder in self.sub_folders:
            num += len(os.listdir(folder))
        return num
    
    def __getitem__(self, idx):
        files = os.listdir(self.current_folder) 
        index = idx

        for i in range(0, self.seen_folder):
            index -= len(os.listdir(os.path.join(self.root_dir, self.sub_folders[i])))
            
        if index == len(files):
            self.seen_folder++
            self.current_folder = os.path.join(self.root_dir, self.sub_folders[self.seen_folders])
            index = 0

        target_file_path = os.path.join(self.current_folder, files[index])
        text = self.reader.read(target_file_path)

        return text, self.seen_folder


class Trainer():
    def __init__(self, root_dir):
        dataset = File_Dataset(root_dir)
        self.loader = Dataloader(dataset, batch_size= 1, shuffle = True)
        model = Doc_network(len(os.listdir(root_dir)))
        self.model = model.to(torch.device("cuda" if torch.cuda_is_available() else "cpu"))

    def train(self, epoch):
        loss_fn = torch.nn.NLLLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum = 0,9)

        for i in range(epoch):
            running_loss = 0.0
            count = 0
            for text, label in self.loader:
                optimizer.zero_grad()
                
                res = torch.argmax(self.model(text))
                
                loss = loss_fn(res, label)
                loss.backward()
                
                optimizer.step()

                running_loss += loss.item()
                count++

            print(running_loss/count)
    
                

        
