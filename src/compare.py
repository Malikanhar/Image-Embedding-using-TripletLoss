import cv2
import argparse
import scipy.spatial
import numpy as np
import tensorflow as tf

def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="prefix")
    return graph

def fixed_image_standardization(image):
    image = image.astype(dtype=np.float32, copy=False)
    np.subtract(image, 127.5, out=image, dtype=np.float32)
    np.multiply(image, 1/128.0, out=image, dtype=np.float32)
    return image

def get_feature (imgs, graph):
    in_img = graph.get_tensor_by_name("prefix/input:0")
    out_feature = graph.get_tensor_by_name("prefix/embeddings:0")
    phase_train = graph.get_tensor_by_name("prefix/phase_train:0")
    for i, img in enumerate(imgs):
        imgs[i] = fixed_image_standardization(img)
    with tf.Session(graph=graph) as sess:
        y_out = sess.run(out_feature, feed_dict={ in_img: imgs , phase_train: False })
    return y_out

def cosine_similarity (featr1, featr2):
    return (1 - np.abs(scipy.spatial.distance.cosine(featr1, featr2)))

def load_image(filename):
  return cv2.resize(cv2.imread(filename), (32, 32))

def main():
    parser = argparse.ArgumentParser(description='Parser for Comparing two images')
    parser.add_argument('--frozen_graph', type=str, required=True,
                                    help='Path to frozen graph with .pb extensions')
    parser.add_argument('--image1', type=str, required=True,
                                    help='Image 1 filename to be compared')
    parser.add_argument('--image2', type=str, required=True,
                                    help='Image 2 filename to be compared')

    args = parser.parse_args()
    graph = load_graph(args.frozen_graph)
    img1 = load_image(args.image1)
    img2 = load_image(args.image2)

    embedding_1, embedding_2 = get_feature([img1, img2], graph)
    similarity = cosine_similarity(embedding_1, embedding_2)

    print('Similarity : {:.2%}'.format(similarity))

if __name__ == "__main__":
    main()