from modules.config_params import *



#############################################################
##### Creating desired data distribution among clients  #####
#############################################################

data_path='../data'
class DatasetHandler:

  def __init__(self, num_clients, batch_size):
    self.num_clients=num_clients
    self.batch_size=batch_size


    # Image augmentation 
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Loading CIFAR10 using torchvision.datasets
    traindata = datasets.CIFAR10(data_path, train=True, download=True,
                          transform= transform_train)

    # Dividing the training data into num_clients, with each client having equal number of images
    traindata_split = torch.utils.data.random_split(traindata, [int(traindata.data.shape[0] / self.num_clients) for _ in range(self.num_clients)])

    # Creating a pytorch loader for a Deep Learning model
    self.train_loader = [torch.utils.data.DataLoader(x, batch_size=self.batch_size, shuffle=True) for x in traindata_split]

    # Normalizing the test images
    self.transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Loading the test iamges and thus converting them into a test_loader
    self.test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(data_path, train=False, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
            ), batch_size=self.batch_size, shuffle=True)
    
  def get_train_loader(self, i):
    return self.train_loader[i]
  
  def get_test_loader(self):
    return self.test_loader

  