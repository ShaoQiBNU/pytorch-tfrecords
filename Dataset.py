###################### load packages ########################
import torch.utils.data
from read_tfrecords import get_data


###################### IrisDataset class ########################
class FlowerDataset(torch.utils.data.Dataset):

    ############ init ###########
    def __init__(self, filenames, data_size, train=True, transform=None):
        self.transform = transform
        self.train = train

        if self.train:
            self.train_data, self.train_label = get_data(filenames, data_size).main()

        else:
            self.test_data, self.test_label = get_data(filenames, data_size).main()


    ############ get data ###########
    def __getitem__(self, index):

        if self.train:
            feature, label = self.train_data[index, :, :, :], self.train_label[index]
        else:
            feature, label = self.test_data[index, :, :, :], self.test_label[index]

        return feature, label


    ############ get data length ###########
    def __len__(self):
        if self.train:
           return len(self.train_data)
        else:
           return len(self.test_data)
