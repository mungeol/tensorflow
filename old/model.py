from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing

import six
import tensorflow as tf
import tensorflow.contrib.layers as lays

CSV_COLUMN_DEFAULTS = [[0.0] for i in range(784)]

def autoencoder(inputs):
    # encoder
    # 28 x 28 x 1   ->  14 x 14 x 32
    # 14 x 14 x 32  ->  7 x 7 x 16
    # 7 x 7 x 16    ->  7 x 7 x 8
    net = lays.conv2d(inputs, 32, [5, 5], stride=2, padding='SAME')
    #print(net.get_shape())
    net = lays.conv2d(net, 16, [5, 5], stride=2, padding='SAME')
    #print(net.get_shape())
    net = lays.conv2d(net, 8, [5, 5], stride=1, padding='SAME')
    #print(net.get_shape())
    # decoder
    # 7 x 7 x 8    ->  7 x 7 x 16
    # 7 x 7 x 16   ->  14 x 14 x 32
    # 14 x 14 x 32  ->  28 x 28 x 1
    net = lays.conv2d_transpose(net, 16, [5, 5], stride=1, padding='SAME')
    #print(net.get_shape())
    net = lays.conv2d_transpose(net, 32, [5, 5], stride=2, padding='SAME')
    #print(net.get_shape())
    net = lays.conv2d_transpose(net, 1, [5, 5], stride=2, padding='SAME', activation_fn=tf.nn.tanh)
    #print(net.get_shape())
    return net

def model_fn(features, mode, params):
    ae_inputs = tf.reshape(features['x'], [-1, 28, 28, 1])
    ae_outputs = autoencoder(ae_inputs)  # create the Autoencoder network

    # calculate the loss and optimize the network
    loss = tf.reduce_mean(tf.square(ae_outputs - ae_inputs))  # claculate the mean square error loss
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "score": loss
        }
        export_outputs = {
            'predict': tf.estimator.export.PredictOutput(predictions)
        }
        
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        'mse':tf.metrics.mean_squared_error(ae_inputs, ae_outputs)
    }
    
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def csv_serving_input_fn():
    csv_row = tf.placeholder(
        shape=[None],
        dtype=tf.string
    )
    
    features = parse_csv(csv_row)
    
    return tf.estimator.export.ServingInputReceiver(features, {'x': csv_row})

SERVING_FUNCTIONS = {
    'CSV': csv_serving_input_fn
}

def parse_csv(rows_string_tensor):

    row_columns = tf.expand_dims(rows_string_tensor, -1)
    columns = tf.decode_csv(row_columns, record_defaults=CSV_COLUMN_DEFAULTS)
    columns = tf.concat(columns, axis=0)
        
    return {'x':columns}

def input_fn(filenames,
                        num_epochs=None,
                        shuffle=True,
                        skip_header_lines=0,
                        batch_size=200):
    filename_dataset = tf.data.Dataset.from_tensor_slices(filenames)
    if shuffle:
      # Process the files in a random order.
      filename_dataset = filename_dataset.shuffle(len(filenames))
      
    # For each filename, parse it into one element per line, and skip the header
    # if necessary.
    dataset = filename_dataset.flat_map(
        lambda filename: tf.data.TextLineDataset(filename).skip(skip_header_lines))
    
    dataset = dataset.map(parse_csv)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size * 10)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    features = iterator.get_next()
    return features

def build_estimator(run_config, hparams):
    estimator = tf.estimator.Estimator(model_fn=model_fn, 
                                  params=hparams, 
                                  config=run_config)
    
    print("")
    print("Estimator Type: {}".format(type(estimator)))
    print("")

    return estimator