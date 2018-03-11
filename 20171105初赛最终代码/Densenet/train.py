import model
import tensorflow as tf
import cv2
import time
import os
import matplotlib.pyplot as plt
import numpy as np


#读取训练集产生tfrecords
def create_record():
    writer = tf.python_io.TFRecordWriter("train.tfrecords")
    input_path = "E:/Tianchi/Densenet/my_network/output_244_all/orig/"
    label_path = "E:/Tianchi/Densenet/my_network/output_244_all/bz/"
    image_number = len(os.listdir(label_path))
    # image_number = 100
    for img_name_index in range(image_number):
        img_path_input = input_path + os.listdir(label_path)[img_name_index]
        img_path_label = label_path + os.listdir(label_path)[img_name_index]
        img_input = cv2.imread(img_path_input)
        # img_input_upsample = cv2.resize(img_input,(upscale*width,upscale*height),interpolation = cv2.INTER_CUBIC)
        img_label = cv2.imread(img_path_label)
        img_label = img_label[:,:,0:1]
        # img_label=img_label[:,:,0:1]
        # img_input_1= img_input[0:112,0:112,:]
        # img_input_2 = img_input[0:112, 112:224, :]
        # img_input_3 = img_input[112:224, 0:112, :]
        # img_input_4 = img_input[112:224, 112:224, :]
        # img_label_1 = img_label[0:112,0:112,:]
        # img_label_2 = img_label[0:112, 112:224, :]
        # img_label_3 = img_label[112:224, 0:112, :]
        # img_label_4 = img_label[112:224, 112:224, :]
        # list_input = [img_input_1,img_input_2,img_input_3,img_input_4]
        # list_label = [img_label_1,img_label_2,img_label_3,img_label_4]
        # for i in range(4):
        #     img_raw_input = list_input[i].tobytes()
        #     img_raw_label = list_label[i].tobytes()
        img_input = img_input.tobytes()
        img_label = img_label.tobytes()
        # img_label = img_label[:,:,0]
        # img_label = img_label-img_input_upsample
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_label])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_input])),
        }))
        writer.write(example.SerializeToString())
    print("tfrecords loaded")
    writer.close()

#读取tfrecords用于训练
def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.string),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })

    img_input = tf.decode_raw(features['img_raw'], tf.uint8)
    img_input = tf.reshape(img_input, [height, width,3])
    img_input = tf.cast(img_input, tf.float32)
    img_input = img_input * (1. / 255)

    img_label = tf.decode_raw(features['label'], tf.uint8)
    img_label = tf.reshape(img_label, [ height, width,1])
    img_label = tf.cast(img_label, tf.float32)
    # img_label[img_label>100]=1
    # img_label[img_label<=100]=0

    img_label=img_label/255
    a=img_label[2,3,0]
    return img_input , img_label


if __name__ =='__main__':
    sess = tf.InteractiveSession()
    height = 224#训练图片的高
    width = 224#训练图片的宽
    batch_size = 16
    write_number=0
    is_training=True
    previous_time = time.clock()
    total_loss_list=[]
    # create_record()
    img , label = read_and_decode("train.tfrecords")
    img_batch , label_batch = tf.train.shuffle_batch([img , label],
                                            batch_size=batch_size, capacity=5000,
                                            min_after_dequeue=1000)
    label_batch = tf.reshape(label_batch, [batch_size, height, width,1])
    img_batch = tf.reshape(img_batch,[batch_size,height,width,3])
    size = np.array([batch_size,height, width])
    annotation_pred= model.inference(img_batch,is_training)
    # annotation_pred = tf.cast(annotation_pred,tf.float32)
    loss = tf.reduce_mean(tf.square(tf.subtract(annotation_pred,label_batch)))

    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(0.0003, global_step,2000, 0.9, staircase=True)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    saver = tf.train.Saver()
    # 初始化所有的op
    sess.run(tf.global_variables_initializer())
    ckpt=tf.train.get_checkpoint_state('E:/Tianchi/my_network/model/')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess,ckpt.model_checkpoint_path)
        print('model restored')
    # 启动队列
    rootPath='E:/Tianchi/Densenet/my_network/'
    for i in range(300000):
        threads = tf.train.start_queue_runners(sess=sess)
        train_step.run()
        loss_now=sess.run([loss])
        total_loss_list.append(loss_now)
        if i % 500 == 0:
            now = time.clock()
            temp = now
            now = now - previous_time
            previous_time = temp
            print("the batch number is %d" % (i))
            print("error is ", loss_now)
            print("the cost time is ", now)
        #每训练1000个batch，画出各个loss变化图
        if i%1000==0 and i!=0:
            plt.plot(total_loss_list)
            plt.xlabel('training times')
            plt.ylabel('total_loss')
            savepath=rootPath+'error_list\\total'+str(i)+'.png'
            plt.savefig(savepath)
            plt.close()
            # total_loss_list = []
        # #每训练50000个batch，保存训练的参数
        if i%5000==0 and i!=0:
            model_root_path=rootPath+"model\\"
            model_path = model_root_path+'model.ckpt'
            saver.save(sess, model_path,i)
        #每训练50000个batch，测试网络的超分辨率结果
        if i%5000==0 and i!=0:
            path=rootPath+'image_test\\'
            write_path=rootPath+'image_test_result\\'
            image_number = len(os.listdir(path))
            write_number=write_number+1
            for img_name_index in range(image_number):
                img_path_input = path + os.listdir(path)[img_name_index]
                img_test = cv2.imread(img_path_input)
                height,width,channel = img_test.shape
                img_test_tensor = tf.convert_to_tensor(img_test, dtype=tf.uint8)
                img_input = tf.reshape(img_test_tensor, [1, height, width, channel])
                img_input = tf.cast(img_input, tf.float32)
                img_input = img_input * (1. / 255)
                size = np.array([1, height, width])
                output = model.inference(img_input, is_training=False,scope_reuse=True)
                # output =model_enhance_subpixel_BN.transform_net(img_input,size, upscale,scope_reuse=True,is_training=False)
                output = output * 255
                output = tf.reshape(output, [height,width, 1])
                output = output.eval()
                output[output>100]=255
                output[output<=100]=0
                savepath=write_path+os.listdir(path)[img_name_index]+'_'+str(write_number)+'.png'
                cv2.imwrite(savepath,output)
    saver.save(sess, "model/model.ckpt")
