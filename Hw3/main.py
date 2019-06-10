from LoadImg import LoadData
from Network import CNN_Net
import numpy as np
from keras.utils import to_categorical


def main():

    data,  label = LoadData('./data/train.csv')
    batch_size = 32
    n_batch = int(len(data))//32

    train_data = []
    train_label = []


    #打乱training data的顺序
    state = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(state)
    np.random.shuffle(label)

    #转换为one-hot编码
    label = to_categorical(label)

    train_batch = int(n_batch*0.7)
    for batch in range(n_batch-1):
        if(batch < train_batch-1):
            train_data.append(data[batch_size*batch: (batch+1)*batch_size-1])
            train_label.append(label[batch_size*batch: (batch+1)*batch_size-1])

    test_data = data[int(n_batch*0.7*batch_size):]
    test_label = label[int(n_batch * 0.7 * batch_size):]

    #print('batch num:', batch)
    train_data = np.array(train_data)
    train_label = np.array(train_label)

    # print('Data: ', train_data)
    # print('Label: ', train_label)
    CNN_Net(train_data, train_label, test_data, test_label, train_batch)
    print(test_data)
  #  print('test_y', test_label[0])
   # print('len： ', len(test_label[0]))
    print('train_data_size: ', train_data.shape)
main()
