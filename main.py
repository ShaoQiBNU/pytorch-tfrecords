###################### load packages ####################
import Dataset
import torch
from PIL import Image
import config

###################### main函数 ####################
def main():
    ########### 读取配置文件 ##########
    ch = config.ConfigHandler("./config.ini")
    ch.load_config()

    ########### 读取参数 ##########
    train_size = int(ch.config["model"]["train_size"])
    train_batch_size = int(ch.config["model"]["train_batch_size"])
    test_batch_size = int(ch.config["model"]["test_batch_size"])
    num_epochs = int(ch.config["model"]["num_epochs"])

    train_file = ch.config["data"]["train_file"]
    test_file = ch.config["data"]["test_file"]

    ########### 获取训练数据loader ##########
    data_train = Dataset.FlowerDataset(train_file, train_size, train=True)
    data_loader_train = torch.utils.data.DataLoader(dataset=data_train, batch_size=train_batch_size, shuffle=True)

    ########### 查看训练数据 ##########
    for epoch in range(0, num_epochs):
        for batch_idx, (data, target) in enumerate(data_loader_train):
            for i in range(0, len(data)):
                im = Image.fromarray(data[i, :, :, :].numpy())
                im.save(str(epoch) + "_train_" + str(batch_idx) + "_" + str(i) + ".jpeg")


    ########### 获取测试数据loader ##########
    data_test = Dataset.FlowerDataset(test_file, test_batch_size, train=False)
    data_loader_test = torch.utils.data.DataLoader(dataset=data_test, batch_size=test_batch_size, shuffle=False)

    ########### 查看测试数据 ##########
    print(type(data_loader_test))
    dataiter = iter(data_loader_test)
    img, label = dataiter.next()
    print(type(img), type(label))
    print(img.size(), label.size())

    for i in range(test_batch_size):
        im = Image.fromarray(img[i, :, :, :].numpy())
        im.save("test_" + str(i) + ".jpeg")

if __name__ == "__main__":
    main()
