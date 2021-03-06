import argparse
import glob
import tensorflow as tf
import multiprocessing as mp
import sys
import os
import logging
import time
import redis

image_classify_redis = redis.Redis(host='localhost', port=6379, db=0)

logger = logging.getLogger(__name__)

# speicherorte fuer trainierten graph und labels in train.sh festlegen ##

# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

image_path = sys.argv[1]
# angabe in console als argument nach dem aufruf  


#bilddatei readen

# holt labels aus file in array 
label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile("tf_files/retrained_labels.txt")]
# !! labels befinden sich jeweils in eigenen lines -> keine aenderung in retrain.py noetig -> falsche darstellung im windows editor !!
           
# graph einlesen, wurde in train.sh -> call retrain.py trainiert
with tf.gfile.FastGFile("tf_files/retrained_graph.pb", 'rb') as f:
  graph_def = tf.GraphDef() ## The graph-graph_def is a saved copy of a TensorFlow graph; objektinitialisierung
  graph_def.ParseFromString(f.read()) #Parse serialized protocol buffer data into variable
  _ = tf.import_graph_def(graph_def, name='') # import a serialized TensorFlow GraphDef protocol buffer, extract objects in the GraphDef as tf.Tensor
  
  #https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/inception.py ; ab zeile 276

print("Model loaded")

def main(input_dir, output_dir):
  start_time = time.time()
  pool = mp.Pool(processes=mp.cpu_count())

  if not os.path.exists(output_dir):
      os.makedirs(output_dir)

  for image_dir in os.listdir(input_dir):
      image_output_dir = os.path.join(output_dir, os.path.basename(os.path.basename(image_dir)))
      if not os.path.exists(image_output_dir):
          os.makedirs(image_output_dir)

  image_paths = glob.glob(os.path.join(input_dir, '**/*.png'))
  for index, image_path in enumerate(image_paths):
      image_output_dir = os.path.join(output_dir, os.path.basename(os.path.dirname(image_path)))
      pool.apply_async(classify_image, (image_path, image_output_dir))

  pool.close()
  pool.join()
  logger.info('Completed in {} seconds'.format(time.time() - start_time))

def classify_image(image_path, image_output_dir):
  print("Processing %s" % image_path)
  with tf.Session() as sess:
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    predictions = sess.run(softmax_tensor, \
             {'DecodeJpeg/contents:0': image_data})

    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

    for node_id in top_k:
      human_string = label_lines[node_id]
      score = predictions[0][node_id]
      if score * 100 < 10:
        human_string = "unknown"
        score = 1
      print('%s (score = %.5f)' % (human_string, score))
      # output_dir = os.path.join(image_output_dir, human_string)
      # output_path = os.path.join(image_output_dir, human_string, "%s_%.2f.png" % (os.path.basename(image_path).replace(".png", ""), score * 100))
      # if not os.path.exists(output_dir):
      #   os.makedirs(output_dir)
      # print("Writing %s" % output_path)
      # f = open(output_path,"wb")
      # f.write(image_data)
      # f.close()
      # print("Written %s" % output_path)
      image_name = os.path.basename(image_path).replace(".png", "")
      image_key = "image:class:"+image_name
      image_value = "%s|%.2f" % (human_string, score * 100)
      image_classify_redis.set(image_key, image_value)
      print("Saved %s to redis (%s=%s)" % (image_name, image_key, image_value))
      break

if __name__ == '__main__':
  parser = argparse.ArgumentParser(add_help=True)
  parser.add_argument('--input-dir', type=str, action='store', default='data', dest='input_dir')
  parser.add_argument('--output-dir', type=str, action='store', default='output', dest='output_dir')
  args = parser.parse_args()
  main(args.input_dir, args.output_dir)
