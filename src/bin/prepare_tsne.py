import tensorflow as tf
import numpy as np

with open('after.npy', 'rb') as f:
    np_data = np.load(f)


embedding_var = tf.Variable(np_data, name='embedding')
# embedding_var = tf.Variable(tf.truncated_normal([10000, 10]), name='embedding')

from tensorflow.contrib.tensorboard.plugins import projector

with tf.Session() as sess:
    # Create summary writer.
    writer = tf.summary.FileWriter('./graphs/embedding_after', sess.graph)
    # Initialize embedding_var
    sess.run(embedding_var.initializer)
    # Create Projector config
    config = projector.ProjectorConfig()
    # Add embedding visualizer
    embedding = config.embeddings.add()
    # Attache the name 'embedding'
    embedding.tensor_name = embedding_var.name
    # Metafile which is described later
    embedding.metadata_path = './after.csv'
    # Add writer and config to Projector
    projector.visualize_embeddings(writer, config)
    # Save the model
    saver_embed = tf.train.Saver([embedding_var])
    saver_embed.save(sess, './graphs/embedding_after/embedding_after.ckpt', 1)

writer.close()