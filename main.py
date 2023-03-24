from modules import config_params as c
from modules import dataset_handling as d
from modules import helper as h
from modules import node as n
import numpy as np
from tqdm import tqdm




def main():
    # Create a command-line argument parser
    # parser = argparse.ArgumentParser(description='My program description')

    # # Add command-line arguments
    # parser.add_argument('--input', type=str, required=True, help='Path to input file')
    # parser.add_argument('--output', type=str, required=True, help='Path to output file')

    # # Parse command-line arguments
    # args = parser.parse_args()
    # args.input
    
    num_clients=20
    epochs=5
    learning_rate=0.1
    classes_pc = 2
    num_selected = 6
    num_rounds = 150
    batch_size = 32
    baseline_num = 100
    retrain_epochs = 20

    config = c.ConfigParams(batch_size=batch_size, epochs=epochs, num_clients=num_clients, num_rounds=num_rounds, num_selected=num_selected)
    datsetHandler = d.DatasetHandler(batch_size=batch_size, num_clients=num_clients)
    helper = h.Helpers()
    nodes = {}
    
    # init client and server nodes
    for id in range(num_clients):
      nodes[id] = n.Node(id=id, model='VGG19', learning_rate=learning_rate)

    srv = n.Node(id=999, model='VGG19', learning_rate=learning_rate)

    loader_fixed = helper.baseline_data(num=baseline_num, datsetHandler=datsetHandler)
    train_loader, test_loader = datsetHandler.get_data_loaders(classes_pc=classes_pc, nclients= num_clients,
                                                      batch_size=batch_size,verbose=True)

    # training loop
    for r in range(config.get_num_rounds()):
      # select random clients
      client_idx = np.random.permutation(config.get_num_clients())[:config.get_num_selected()]
      client_lens = [len(train_loader[idx]) for idx in client_idx]# client update
      
      # client update
      loss = 0
      for i in tqdm(range(config.get_num_selected())):
          helper.client_syn(client_model=nodes[i].get_model(), global_model=srv.get_model())
          loss += helper.client_update(client_model=nodes[i].get_model(), optimizer=nodes[i].get_opt(), train_loader=train_loader[client_idx[i]], epoch=epochs)
      
      config.losses_train_append(loss)
      # server aggregate
      #### retraining on the global server
      loss_retrain =0
      for i in tqdm(range(num_selected)):
        loss_retrain+= helper.client_update(client_model=nodes[i].get_model(), optimizer=nodes[i].get_opt(), train_loader=loader_fixed, epoch=retrain_epochs)
      config.losses_retrain_append(loss_retrain)
      
      ### Aggregating the models
      helper.server_aggregate(srv.get_model(), [nodes[id].get_model() for id in range(num_clients)], client_lens)
      
      test_loss, acc = helper.test(global_model=srv.get_model(), test_loader=test_loader)
      config.losses_test_append(test_loss)
      config.acc_test_append(acc)
      print('%d-th round' % r)
      print('average train loss %0.3g | test loss %0.3g | test acc: %0.3f' % (loss / config.get_num_selected(), test_loss, acc))


def main_test():
  print('ktm')
  datsetHandler = d.DatasetHandler(num_clients=20, batch_size=32)
  helper = h.Helpers()
  loader = helper.baseline_data(num=100, datsetHandler=datsetHandler)


if __name__ == '__main__':
    main()


