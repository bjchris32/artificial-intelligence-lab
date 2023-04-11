# submitted.py

"""
This is the module you will submit to the autograder.

There are several function and variable definitions, here, that raise RuntimeErrors.
You should replace each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.

If you are not sure how to use PyTorch, you may want to take a look at the tutorial.
"""

import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from models import resnet18


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


"""
1.  Define and build a PyTorch Dataset
"""
class CIFAR10(Dataset):
    def convert_image_label(self, dataset):
        pair_list = []
        label_list = dataset[b'labels']
        data_list = dataset[b'data']

        # convert 1d array to tensor
        # ref: https://stackoverflow.com/questions/58778364/convert-a-cifar-1d-array-from-pickle-to-an-image-rgb
        img_list = []
        for data in data_list:
            tmp_data = np.array(data)
            img = tmp_data.reshape(3, 32, 32)
            img_view = np.transpose(img, (1, 2, 0))
            tensor = img_view
            # TODO: change the transform function based on test or training
            if self.transform:
                tensor = self.transform(img_view)
            img_list.append(tensor)

        for pair in zip(img_list, label_list):
            pair_list.append(pair)

        return pair_list


    def __init__(self, data_files, transform=None, target_transform=None):
        """
        Initialize your dataset here. Note that transform and target_transform
        correspond to your data transformations for train and test respectively.
        """

        # airplane (0), automobile (1), bird (2), deer (3), frog (4), horse (5), ship (6), truck (7)

        # transform: train
        # target_transform: test

        # iterate parameters
        # @ set require False

        # pickles: data_batch_1, data_batch_2, data_batch_3, data_batch_4, data_batch_5, test_batch

        self.transform = transform
        self.target_transform = target_transform
        img_data_pairs = []
        for data_file in data_files:
            unpickle_dataset = unpickle(data_file)
            img_data_pairs += self.convert_image_label(unpickle_dataset)

        # should be a list of [image, label]
        self.preprocessed_data = img_data_pairs

    def __len__(self):
        """
        Return the length of your dataset here.
        """
        return len(self.preprocessed_data)


    def __getitem__(self, idx):
        """
        Obtain a sample from your dataset. 

        Parameters:
            x:      an integer, used to index into your data.

        Outputs:
            y:      a tuple (image, label), although this is arbitrary so you can use whatever you would like.
        """
        print("__getitem__ ======= ")
        # img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # image = read_image(img_path)
        # label = self.img_labels.iloc[idx, 1]
        # if self.transform:
        #     image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)

        image, label = self.preprocessed_data[idx]
        return image, label
        


        # raise NotImplementedError("You need to write this part!")
    

def get_preprocess_transform(mode):
    """
    Parameters:
        mode:           "train" or "test" mode to obtain the corresponding transform
    Outputs:
        transform:      a torchvision transforms object e.g. transforms.Compose([...]) etc.
    """
    input_size = 224
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        # transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip()])
    return data_transforms


def build_dataset(data_files, transform=None):
    """
    Parameters:
        data_files:      a list of strings e.g. "cifar10_batches/data_batch_1" corresponding to the CIFAR10 files to load data
        transform:       the preprocessing transform to be used when loading a dataset sample
    Outputs:
        dataset:      a PyTorch dataset object to be used in training/testing
    """

    print("build_dataset build_dataset build_dataset dataset = ", data_files)

    cifar10_dataset = CIFAR10(data_files, transform=transform)

    return cifar10_dataset.preprocessed_data


# Q: how to load test set?
"""
2.  Build a PyTorch DataLoader
"""
def build_dataloader(dataset, loader_params):
    """
    Parameters:
        dataset:         a PyTorch dataset to load data
        loader_params:   a dict containing all the parameters for the loader. 
        
    Please ensure that loader_params contains the keys "batch_size" and "shuffle" corresponding to those 
    respective parameters in the PyTorch DataLoader class. 

    Outputs:
        dataloader:      a PyTorch dataloader object to be used in training/testing
    """

    return DataLoader(dataset, batch_size=loader_params['batch_size'], shuffle=loader_params['shuffle'])


"""
3. (a) Build a neural network class.
"""
# which layers to use as your backbone
# (hint: only the final part of your network should be excluded from your backbone).
# What is backbone?

# Q: What is the difference between input size (224, 224) and input feature 512?
# Q: should we change the parameter of init, so that we could pass the resnet model with checkpoint?
# -> no
# Q: load checkpoint after initializing the model, or initializing the model after loading checkpoint?
# same time
# Q: iterate the datafile and get the the best trained model, or train with all data?
# Q: where to implement/control test?
# Q: where to use target transform?

class FinetuneNet(torch.nn.Module):
    def __init__(self):
        """
        Initialize your neural network here. Remember that you will be performing finetuning
        in this network so follow these steps:
        
        1. Initialize convolutional backbone with pretrained model parameters.
        2. Freeze convolutional backbone.
        3. Initialize linear layer(s). 
        """
        super().__init__()
        ################# Your Code Starts Here #################
        num_classes = 8
        RESTNET_PATH = 'resnet18.pt'
        checkpoint = torch.load(RESTNET_PATH)
        # print("len(checkpoint) = ", len(checkpoint))
        # self.model = resnet18(pretrained = True)
        self.model = resnet18()
        
        self.model.load_state_dict(checkpoint)
        print("self.model.parameters() = ", self.model.parameters())
        for param in self.model.parameters():
            # print("param = ", param)
            param.requires_grad = False
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

        # for name, param in self.model.named_parameters():
        #     print("name ", name, "param.requires_grad = ", param.requires_grad)

        
        # named_layers = dict(model.named_modules())
        # print("named_layers.keys() = ", named_layers.keys())

        


        ################## Your Code Ends here ##################

    def forward(self, x):
        """
        Perform a forward pass through your neural net.

        Parameters:
            x:      an (N, input_size) tensor, where N is arbitrary.

        Outputs:
            y:      an (N, output_size) tensor of output from the network
        """
        ################# Your Code Starts Here #################
        # reshape 2d to 3d before passing into cnn
        # print("before reshape, x size:", x.size(), "\n")
        # before reshape, x size: torch.Size([100, 2883]) 

        return self.model(x)

        ################## Your Code Ends here ##################


"""
3. (b)  Build a model
"""
def build_model(trained=False):
    """
    Parameters:
        trained:         a bool value specifying whether to use a model checkpoint

    Outputs:
        model:           the model to be used for training/testing
    """    
    net = FinetuneNet()
    return net.model


"""
4.  Build a PyTorch optimizer
"""
# Q: where should we specify the test?
def build_optimizer(optim_type, model_params, hparams):
    """
    Parameters:
        optim_type:      the optimizer type e.g. "Adam" or "SGD"
        model_params:    the model parameters to be optimized
        hparams:         the hyperparameters (dict type) for usage with learning rate 

    Outputs:
        optimizer:       a PyTorch optimizer object to be used in training
    """
    if optim_type == 'SGD':
        optimizer = torch.optim.SGD(params=model_params, lr=0.1)
    else:
        optimizer = torch.optim.Adam(params=model_params, lr=0.1)
    
    return optimizer


"""
5. Training loop for model
"""
def train(train_dataloader, model, loss_fn, optimizer):
    """
    Train your neural network.

    Iterate over all the batches in dataloader:
        1.  The model makes a prediction.
        2.  Calculate the error in the prediction (loss).
        3.  Zero the gradients of the optimizer.
        4.  Perform backpropagation on the loss.
        5.  Step the optimizer.

    Parameters:
        train_dataloader:   a dataloader for the training set and labels
        model:              the model to be trained
        loss_fn:            loss function
        optimizer:          optimizer
    """

    ################# Your Code Starts Here #################
    for features, labels in train_dataloader:
        y_pred = model(features)
        loss = loss_fn(y_pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    # raise NotImplementedError("You need to write this part!")
    ################## Your Code Ends here ##################


"""
6. Testing loop for model
"""
def test(test_dataloader, model):
    """
    This part is optional.

    You can write this part to monitor your model training process.

    Test your neural network.
        1.  Make sure gradient tracking is off, since testing set should only
            reflect the accuracy of your model and should not update your model.
        2.  The model makes a prediction.
        3.  Calculate the error in the prediction (loss).
        4.  Print the loss.

    Parameters:
        test_dataloader:    a dataloader for the testing set and labels
        model:              the model that you will use to make predictions


    Outputs:
        test_acc:           the output test accuracy (0.0 <= acc <= 1.0)
    """
    model.eval()

    loss_fn = torch.nn.CrossEntropyLoss()

    running_loss = 0.0
    running_corrects = 0

    for features, labels in test_dataloader:
        y_pred = model(features)
        loss = loss_fn(y_pred, labels)
        _, preds = torch.max(y_pred, 1)
        # optimizer.zero_grad()
        running_loss += loss.item() * features.size(0)
        running_corrects += torch.sum(preds == labels.data)
        
    epoch_loss = running_loss / len(test_dataloader)
    epoch_acc = running_corrects.double() / len(test_dataloader)
    print('{} Loss: {:.4f} Acc: {:.4f}'.format('test', epoch_loss, epoch_acc))

    # test_loss = something
    # print("Test loss:", test_loss)
    # raise NotImplementedError("You need to write this part!")

"""
7. Full model training and testing
"""
def run_model():
    """
    The autograder will call this function and measure the accuracy of the returned model.
    Make sure you understand what this function does.
    Do not modify the signature of this function (names and parameters).

    Please run your full model training and testing within this function.

    Outputs:
        model:              trained model
    """
    training_data_files = ["cifar10_batches/data_batch_1", "cifar10_batches/data_batch_2", "cifar10_batches/data_batch_3", "cifar10_batches/data_batch_4", "cifar10_batches/data_batch_5"]
    # training_data_files = ["cifar10_batches/data_batch_1"]
    
    model = build_model(trained=True)

    # Q: Should we take the loss function from the checkpoint?
    # Q: Should we update the state of optimizer from the checkpoint?
    loss_fn = torch.nn.CrossEntropyLoss()
    optim_type = 'Adam'
    model_params = model.parameters()
    # print("model_params = ", model_params)
    hparams = {}
    optimizer = build_optimizer(optim_type, model_params, hparams)


    data_transforms = get_preprocess_transform('training')

    training_dataset = build_dataset(training_data_files, transform=data_transforms)
    loader_params = { "batch_size": 8, "shuffle": True }
    train_dataloader = build_dataloader(training_dataset, loader_params)
    train(train_dataloader, model, loss_fn, optimizer)

    return model


def test_model(trained_model):
    testing_data_files = ["cifar10_batches/test_batch"]
    data_transforms = get_preprocess_transform('test')
    
    testing_dataset = build_dataset(testing_data_files, transform=data_transforms)
    loader_params = { "batch_size": 8, "shuffle": True }
    testing_dataloader = build_dataloader(testing_dataset, loader_params)
    test(testing_dataloader, trained_model)
