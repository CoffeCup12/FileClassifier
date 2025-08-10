from model import Doc_network
import torch 
from torch.utils.data import Dataset, DataLoader 
import os
from file_reader import Reader 

class File_Dataset(Dataset):
    def __init__(self, root_dir):
        self.subfolders = os.listdir(root_dir)
        self.root_dir = root_dir
        self.meta_datas = self.load_all_files()
        self.reader = Reader()

    def __len__(self):
        return len(self.meta_datas)
    
    def __getitem__(self, idx):
        meta_data = self.meta_datas[idx]
        text = self.reader.read(meta_data["path"])
        if text == "unclassified":
            label = len(self.subfolders)
        else:
            label = meta_data["label"]
        return text, label
        

    def load_all_files(self):
        dataset = []
        for i, folder in enumerate(self.subfolders):
            folder_path = os.path.join(self.root_dir, folder)
            for file in os.listdir(folder_path):
                dataset.append({"path" : os.path.join(folder_path, file), "label" : i})
        return dataset
                



class Trainer():
    def __init__(self, root_dir):
        dataset = File_Dataset(root_dir)
        self.loader = DataLoader(dataset, batch_size=32, shuffle = True)
        model = Doc_network(len(os.listdir(root_dir))+1)
        self.model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def train(self, epoch):
        loss_fn = torch.nn.NLLLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum = 0.9)

        for i in range(epoch):
            running_loss = 0.0
            count = 0
            for text, label in self.loader:
                optimizer.zero_grad()

                output = self.model(text)

                loss = loss_fn(output, label)
                loss.backward()
                
                optimizer.step()

                running_loss += loss.item()
                count+=1

            print(running_loss/count)

        torch.save(self.model.state_dict(), "model.pth")
    
                

        
