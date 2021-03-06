import paddle.v2 as paddle
import paddle.fluid as fluid
import os
import reader
import numpy as np
import load_model as load_model
from mobilenet_ssd import mobile_net


def train(train_file_list,
          val_file_list,
          data_args,
          learning_rate,
          batch_size,
          num_passes,
          model_save_dir='model',
          init_model_path=None):
    image_shape = [3, data_args.resize_h, data_args.resize_w]

    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    gt_box = fluid.layers.data(
        name='gt_box', shape=[4], dtype='float32', lod_level=1)
    gt_label = fluid.layers.data(
        name='gt_label', shape=[1], dtype='int32', lod_level=1)
    difficult = fluid.layers.data(
        name='gt_difficult', shape=[1], dtype='int32', lod_level=1)

    mbox_locs, mbox_confs, box, box_var = mobile_net(image, image_shape)
    nmsed_out = fluid.layers.detection_output(
        mbox_locs, mbox_confs, box, box_var, nms_threshold=0.45)
    loss_vec = fluid.layers.ssd_loss(mbox_locs, mbox_confs, gt_box, gt_label,
                                     box, box_var)
    loss = fluid.layers.nn.reduce_sum(loss_vec)

    map_eval = fluid.evaluator.DetectionMAP(
        nmsed_out,
        gt_label,
        gt_box,
        difficult,
        21,
        overlap_threshold=0.5,
        evaluate_difficult=False,
        ap_version='11point')
    map, accum_map = map_eval.get_map_var()

    test_program = fluid.default_main_program().clone(for_test=True)
    with fluid.program_guard(test_program):
        test_program = fluid.io.get_inference_program([loss, map, accum_map])

    optimizer = fluid.optimizer.DecayedAdagrad(
        learning_rate=fluid.layers.exponential_decay(
            learning_rate=learning_rate,
            decay_steps=40000,
            decay_rate=0.1,
            staircase=True),
        regularization=fluid.regularizer.L2Decay(0.0005), )

    opts = optimizer.minimize(loss)

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    #load_model.load_and_set_vars(place)
    train_reader = paddle.batch(
        reader.train(data_args, train_file_list), batch_size=batch_size)
    test_reader = paddle.batch(
        reader.test(data_args, val_file_list), batch_size=batch_size)
    feeder = fluid.DataFeeder(
        place=place, feed_list=[image, gt_box, gt_label, difficult])

    def test(pass_id):
        map_eval.reset(exe)
        test_map = None
        for _, data in enumerate(test_reader()):
            test_map = exe.run(test_program,
                               feed=feeder.feed(data),
                               fetch_list=[accum_map])
        print("Test {0}, map {1}".format(pass_id, test_map[0]))

    #print fluid.default_main_program()
    for pass_id in range(num_passes):
        map_eval.reset(exe)
        for batch_id, data in enumerate(train_reader()):
            loss_v, map_v, accum_map_v = exe.run(
                fluid.default_main_program(),
                feed=feeder.feed(data),
                fetch_list=[loss, map, accum_map])
            print(
                "Pass {0}, batch {1}, loss {2}, cur_map {3}, map {4}"
                .format(pass_id, batch_id, loss_v[0], map_v[0], accum_map_v[0]))
        test(pass_id)

        if pass_id % 10 == 0:
            model_path = os.path.join(model_save_dir, str(pass_id))
            print 'save models to %s' % (model_path)
            fluid.io.save_inference_model(model_path, ['image'], [nmsed_out],
                                          exe)


if __name__ == '__main__':
    data_args = reader.Settings(
        data_dir='./data',
        label_file='label_list',
        resize_h=300,
        resize_w=300,
        mean_value=[127.5, 127.5, 127.5])
    train(
        train_file_list='./data/trainval.txt',
        val_file_list='./data/test.txt',
        data_args=data_args,
        learning_rate=0.001,
        batch_size=32,
        num_passes=300)
