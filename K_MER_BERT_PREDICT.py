import sys, getopt
import re
import torch
import numpy as np
import tensorflow as tf
from collections import Counter
from transformers import AutoModel, T5Tokenizer, T5Model
from transformers import AutoTokenizer, AutoModelForMaskedLM, FeatureExtractionPipeline
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("loading ESM-2...")
model_esm2, alphabet_esm2 = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")

def slice_sequence(sequence, window_size=50):
    return [sequence[i:i + window_size] for i in range(0, len(sequence) - window_size + 1, 1)]

kmer = []
def get_kmers(sequence, k=3):
    kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    kmer = kmers
    return Counter[kmers]

def kmer_vector(counter, all_kmers):
    return [counter[kmer] for kmer in all_kmers]

def encode_by_esm2(seq):
    batch_converter = alphabet_esm2.get_batch_converter()
    model_esm2.eval()  
    data = [
        ("tmpDeepLCR", seq.upper()),
    ]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    with torch.no_grad():
        results = model_esm2(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]
    token_representations = token_representations.detach().cpu().numpy()[0][1:-1, :]

    maxsize = 500
    embedding = 400

    res = token_representations[:maxsize, :embedding]
    res = np.pad(res, ((0, 500 - len(res)), (0, 0)), constant_values=0)
    return res

def encode(fasta_file):
    with open(fasta_file) as f:
        lines = f.readlines()
        seq_name = lines[0].strip().replace(">", "")
        seq = lines[-1].strip()
    
    kmer_features = get_kmers(seq, k=3)
    kmer_features_vector = np.array(kmer_vector(kmer_features, kmer))

    features = np.concatenate([encode_by_esm2(seq),
                              kmer_features_vector], axis=-1)

    return features[np.newaxis, :, :], seq_name, len(seq)


def main():
    filelist = ''
    save_path = ''
    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv, "i:o:")
    for opt, arg in opts:
        if opt == '-i':
            filelist = arg
        elif opt == '-o':
            save_path = arg

    if not os.path.exists(filelist):
        print("filelist:{} not found".format(filelist))
        return

    Deep_DL = tf.keras.models.load_model('lib/intergrated_model') 
    with open(filelist) as f:
        for line in f.readlines():
            fasta_file = line.strip()

            if not os.path.exists(fasta_file):
                print("fasta file:{} not found".format(fasta_file))
                continue

            features, id, length = encode(fasta_file)
            pred = Deep_DL.predict(features)
            pred = np.squeeze(pred)[:length]
            np.save(save_path + "/{}_result.npy".format(id), pred)

    print("finish")


main()
