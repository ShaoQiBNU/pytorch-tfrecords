###################### load packages ####################
import Dataset
import torch
from PIL import Image


###################### main函数 ####################
def main():

    ########### 获取训练数据loader ##########
    train_file = "flower_train.tfrecords"
    train_batch_size = 20
    data_train = Dataset.FlowerDataset(train_file, train_batch_size, train=True)
    data_loader_train = torch.utils.data.DataLoader(dataset=data_train, batch_size=train_batch_size, shuffle=True)

    ########### 查看训练数据 ##########
    print(type(data_loader_train))
    dataiter = iter(data_loader_train)
    img, label = dataiter.next()
    print(type(img), type(label))
    print(img.size(), label.size())
    for i in range(0, train_batch_size):
        im = Image.fromarray(img[i, :, :, :].numpy())
        im.save("train_" + str(i) + ".jpeg")

    ########### 获取测试数据loader ##########
    test_file = "flower_test.tfrecords"
    test_batch_size = 10
    data_test = Dataset.FlowerDataset(test_file, test_batch_size, train=False)
    data_loader_test = torch.utils.data.DataLoader(dataset=data_test, batch_size=test_batch_size, shuffle=False)

    ########### 查看测试数据 ##########
    print(type(data_loader_test))
    dataiter = iter(data_loader_train)
    img, label = dataiter.next()
    print(type(img), type(label))
    print(img.size(), label.size())

    for i in range(test_batch_size):
        im = Image.fromarray(img[i, :, :, :].numpy())
        im.save("test_" + str(i) + ".jpeg")

if __name__ == "__main__":
    main()
