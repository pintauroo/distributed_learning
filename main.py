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

    config = c.ConfigParams(batch_size=32, epochs= 5, num_clients=20, num_rounds= 150, num_selected=6)
    datsetHandler = d.DatasetHandler(batch_size=32, num_clients=20)
    helper = h.Helpers()
    nodes = {}
    
    # init client and server nodes
    for id in range(num_clients):
      nodes[id] = n.Node(id=id, model='VGG19')

    srv = n.Node(id=999, model='VGG19')


    # training loop
    for r in range(config.get_num_rounds()):
      # select random clients
      client_idx = np.random.permutation(config.get_num_clients())[:config.get_num_selected()]
      # client update
      loss = 0
      for i in tqdm(range(config.get_num_selected())):
          loss += helper.client_update(client_model=nodes[i].get_model(), optimizer=nodes[i].get_opt(), train_loader=datsetHandler.get_train_loader(client_idx[i]), epoch=epochs)
      
      config.losses_train_append(loss)
      # server aggregate
      helper.server_aggregate(srv.get_model(), [nodes[id].get_model() for id in range(num_clients)])
      
      test_loss, acc = helper.test(srv.get_model(), datsetHandler.get_test_loader())
      config.losses_test_append(test_loss)
      config.acc_test_append(acc)
      print('%d-th round' % r)
      print('average train loss %0.3g | test loss %0.3g | test acc: %0.3f' % (loss / config.get_num_selected(), test_loss, acc))

if __name__ == '__main__':
    main()


