import model
import tensorflow as tf
import time
import os
import numpy as np
import argparse



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


def average_gradients(tower_grads):

  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    grads = []
    for g, _ in grad_and_vars:
      expanded_g = tf.expand_dims(g, 0)
      grads.append(expanded_g)
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--buckets', type=str, default='',help='input data path')
    parser.add_argument('--checkpointDir', type=str, default='',help='output model path')
    FLAGS, _ = parser.parse_known_args()
    sess = tf.InteractiveSession()
    height = 224
    width = 224
    batch_size = 16
    write_number=0
    is_training=True
    previous_time = time.clock()
    total_loss_list=[]
    # create_record()
    records_path = os.path.join('cached://',FLAGS.buckets,"train.tfrecords" )
    img , label = read_and_decode(records_path)
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
    learning_rate = tf.train.exponential_decay(0.001, global_step,10000, 0.8, staircase=True)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    # ckpt=tf.train.get_checkpoint_state('E:/Tianchi/my_network/model/')
    # if ckpt and ckpt.model_checkpoint_path:
    #     saver.restore(sess,ckpt.model_checkpoint_path)
    #     print('model restored')
    # rootPath='E:/Tianchi/Densenet/my_network/'
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
            #         saver.save(sess, model_path,i)
        if i%5000==0 and i!=0:
            model_path = os.path.join(FLAGS.checkpointDir, "model.ckpt")
            saver.save(sess, model_path,i)
    #     if i%1000==0 and i!=0:
    #         plt.plot(total_loss_list)
    #         plt.xlabel('training times')
    #         plt.ylabel('total_loss')
    #         savepath=rootPath+'error_list\\total'+str(i)+'.png'
    #         plt.savefig(savepath)
    #         plt.close()
    #         # total_loss_list = []
    
    #     if i%5000==0 and i!=0:
    #         model_root_path=rootPath+"model\\"
    #         model_path = model_root_path+'model.ckpt'
    #     if i%5000==0 and i!=0:
    #         path=rootPath+'image_test\\'
    #         write_path=rootPath+'image_test_result\\'
    #         image_number = len(os.listdir(path))
    #         write_number=write_number+1
    #         for img_name_index in range(image_number):
    #             img_path_input = path + os.listdir(path)[img_name_index]
    #             img_test = cv2.imread(img_path_input)
    #             height,width,channel = img_test.shape
    #             img_test_tensor = tf.convert_to_tensor(img_test, dtype=tf.uint8)
    #             img_input = tf.reshape(img_test_tensor, [1, height, width, channel])
    #             img_input = tf.cast(img_input, tf.float32)
    #             img_input = img_input * (1. / 255)
    #             size = np.array([1, height, width])
    #             output = model.inference(img_input, is_training=False,scope_reuse=True)
    #             # output =model_enhance_subpixel_BN.transform_net(img_input,size, upscale,scope_reuse=True,is_training=False)
    #             output = output * 255
    #             output = tf.reshape(output, [height,width, 1])
    #             output = output.eval()
    #             output[output>100]=255
    #             output[output<=100]=0
    #             savepath=write_path+os.listdir(path)[img_name_index]+'_'+str(write_number)+'.png'
    #             cv2.imwrite(savepath,output)
    # saver.save(sess, "model/model.ckpt")
