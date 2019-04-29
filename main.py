###################### load packages ####################
import Dataset


###################### main函数 ####################
def main():

    ########### 获取数据loader ##########
    train_file = "flower_train.tfrecords"
    train_batch_size = 20
    data_loader = Dataset.FlowerDataset(train_file, train_batch_size, train=True)
    data_loader_train =data_loader.load_train_data()


    test_file = "flower_test.tfrecords"
    test_batch_size = 10
    data_loader = Dataset.FlowerDataset(test_file, test_batch_size, train=False)
    data_loader_test = data_loader.load_test_data()


if __name__ == "__main__":
    main()