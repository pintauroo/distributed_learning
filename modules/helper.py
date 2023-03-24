from modules.config_params import *


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
        return loss.item()

    def server_aggregate(self, global_model, client_models):
        """
        This function has aggregation method 'mean'
        """
        ### This will take simple mean of the weights of models ###
        global_dict = global_model.state_dict()
        for k in global_dict.keys():
            global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0).mean(0)
        global_model.load_state_dict(global_dict)
        for model in client_models:
            model.load_state_dict(global_model.state_dict())

    def test(self, global_model, test_loader):
        """This function test the global model on test data and returns test loss and test accuracy """
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

        return test_loss, acc
