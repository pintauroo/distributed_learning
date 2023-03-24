from modules.config_params import *
from modules import dataset_handling as d


class Helpers:
    def __init__(self):
        pass


    def client_update(self, client_model, optimizer, train_loader, epoch=5):
        """
        This function updates/trains client model on client data
        """
        client_model.train()
        for e in range(epoch):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.cuda(), target.cuda()
                optimizer.zero_grad()
                output = client_model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
                # if batch_idx % 10 == 0:
                #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #         epoch, batch_idx * len(data), len(train_loader.dataset),
                #         100. * batch_idx / len(train_loader), loss.item()))

        return loss.item()

    
    def server_aggregate(global_model, client_models,client_lens):
        """
        This function has aggregation method 'wmean'
        wmean takes the weighted mean of the weights of models
        """
        total = sum(client_lens)
        n = len(client_models)
        global_dict = global_model.state_dict()
        for k in global_dict.keys():
            global_dict[k] = torch.stack([client_models[i].state_dict()[k].float()*(n*client_lens[i]/total) for i in range(len(client_models))], 0).mean(0)
        global_model.load_state_dict(global_dict)
        for model in client_models:
            model.load_state_dict(global_model.state_dict())


    def test(self, global_model, test_loader):
        """
        This function test the global model on test data 
        and returns test loss and test accuracy 
        """
        global_model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.cuda(), target.cuda()
                output = global_model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        acc = correct / len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

        return test_loss, acc
    
    def baseline_data(self, num, datsetHandler):
        '''
        Returns baseline data loader to be used on retraining on global server
        Input:
                num : size of baseline data
        Output:
                loader: baseline data loader
        '''

        x_train, y_train, x_test, y_test = datsetHandler.get_cifar10()
        x , y = datsetHandler.shuffle_list_data(x_train, y_train)

        x, y = x[:num], y[:num]
        transform, _ = datsetHandler.get_default_data_transforms(train=True, verbose=False)
        loader = torch.utils.data.DataLoader(d.CustomImageDataset(x, y, transform), batch_size=16, shuffle=True)

        return loader
    
    def client_syn(self, client_model, global_model):
        '''
        This function synchronizes the client model with global model
        '''
        client_model.load_state_dict(global_model.state_dict())
