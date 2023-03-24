from modules.VGG_NN import VGG

import torch.optim as optim

class Node:
    def __init__(self, id, model, learning_rate) -> None:
        self.id = id
        if model == 'VGG19':
            self.model = VGG('VGG19').cuda() 
            self.opt = optim.SGD(self.model.parameters(), lr=learning_rate)
            self.model.weights_init()
            

    # def init_model(self):
        # self.model.load_state_dict(self.global_model.state_dict()) ### initial synchronizing with global model 
    def get_model(self):
        return self.model
    
    def get_opt(self):
        return self.opt