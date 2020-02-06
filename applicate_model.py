import tensorflow as tf

def distinguish(img):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver

    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.import_meta_graph('model/mnist_model.ckpt.meta')  # 载入模型结构
        saver.restore(sess, 'model/mnist_model.ckpt')  # 载入模型参数

        graph = tf.get_default_graph()  # 加载计算图
        x = graph.get_tensor_by_name("x:0")  # 从模型中读取占位符变量
        keep_prob = graph.get_tensor_by_name("keep_prob:0")
        y_conv = graph.get_tensor_by_name("model_output:0")  # 关键的一句  从模型中读取占位符变量
        prediction = tf.argmax(y_conv, 1)
        predint = prediction.eval(feed_dict={x: img,keep_prob: 1.0}, session=sess)  # feed_dict输入数据给placeholder占位符

        return predint[0]

