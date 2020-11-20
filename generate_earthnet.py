from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import errno
import json
import os
import random

import cv2
import numpy as np
import tensorflow as tf

from video_prediction import datasets, models
from video_prediction.utils.ffmpeg_gif import save_gif


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="a directory containing the root dat directory ")
    parser.add_argument("--output_dir", type=str, default='outputs', help="directory to save raw outputs from predictions")
    parser.add_argument("--results_dir", type=str, default='results', help="ignored if output_gif_dir is specified")
    parser.add_argument("--results_gif_dir", type=str, help="default is results_dir. ignored if output_gif_dir is specified")
    parser.add_argument("--results_png_dir", type=str, help="default is results_dir. ignored if output_png_dir is specified")
    parser.add_argument("--output_gif_dir", help="output directory where samples are saved as gifs. default is "
                                                 "results_gif_dir/model_fname")
    parser.add_argument("--output_png_dir", help="output directory where samples are saved as pngs. default is "
                                                 "results_png_dir/model_fname")
    parser.add_argument("--checkpoint", help="directory with checkpoint or checkpoint name (e.g. checkpoint_dir/model-200000)")

    parser.add_argument("--mode", type=str, choices=['val', 'test'], default='val', help='mode for dataset, val or test.')

    parser.add_argument("--dataset", type=str, help="dataset class name")
    parser.add_argument("--dataset_hparams", type=str, help="a string of comma separated list of dataset hyperparameters")
    parser.add_argument("--model", type=str, help="model class name")
    parser.add_argument("--model_hparams", type=str, help="a string of comma separated list of model hyperparameters")
    parser.add_argument("--context_length", type=int, help="number of context frames")
    parser.add_argument("--prediction_length", type=int, help="number of frames to predict")

    parser.add_argument("--batch_size", type=int, default=5, help="number of samples in batch")
    parser.add_argument("--num_samples", type=int, help="number of samples in total (all of them by default)")
    parser.add_argument("--num_epochs", type=int, default=1)

    parser.add_argument("--num_stochastic_samples", type=int, default=5)
    parser.add_argument("--gif_length", type=int, help="default is sequence_length")
    parser.add_argument("--fps", type=int, default=4)

    parser.add_argument("--gpu_mem_frac", type=float, default=0, help="fraction of gpu memory to use")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.seed is not None:
        tf.set_random_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
    
    args.results_dir = args.output_dir
    args.results_gif_dir = args.output_dir
    args.results_png_dir = args.output_dir
    dataset_hparams_dict = {}
    model_hparams_dict = {}
    if args.checkpoint:
        checkpoint_dir = os.path.normpath(args.checkpoint)
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), checkpoint_dir)
        if not os.path.isdir(args.checkpoint):
            checkpoint_dir, _ = os.path.split(checkpoint_dir)
        with open(os.path.join(checkpoint_dir, "options.json")) as f:
            print("loading options from checkpoint %s" % args.checkpoint)
            options = json.loads(f.read())
            args.dataset = args.dataset or options['dataset']
            args.model = args.model or options['model']
        try:
            with open(os.path.join(checkpoint_dir, "dataset_hparams.json")) as f:
                dataset_hparams_dict = json.loads(f.read())
        except FileNotFoundError:
            print("dataset_hparams.json was not loaded because it does not exist")
        try:
            with open(os.path.join(checkpoint_dir, "model_hparams.json")) as f:
                model_hparams_dict = json.loads(f.read())
                model_hparams_dict.pop('num_gpus', None)  # backwards-compatibility
        except FileNotFoundError:
            print("model_hparams.json was not loaded because it does not exist")
    else:
        if not args.dataset:
            raise ValueError('dataset is required when checkpoint is not specified')
        if not args.model:
            raise ValueError('model is required when checkpoint is not specified')
        args.output_gif_dir = args.output_gif_dir or os.path.join(args.results_gif_dir, 'model.%s' % args.model)
        args.output_png_dir = args.output_png_dir or os.path.join(args.results_png_dir, 'model.%s' % args.model)

    print('----------------------------------- Options ------------------------------------')
    for k, v in args._get_kwargs():
        print(k, "=", v)
    print('------------------------------------- End --------------------------------------')

    VideoDataset = datasets.get_dataset_class(args.dataset)
    dataset = VideoDataset(args.input_dir, mode=args.mode, num_epochs=args.num_epochs, seed=args.seed,
                           hparams_dict=dataset_hparams_dict, hparams=args.dataset_hparams)

    def override_hparams_dict(dataset):
        hparams_dict = dict(model_hparams_dict)
        hparams_dict['context_frames'] = dataset.hparams.context_frames
        hparams_dict['sequence_length'] = dataset.hparams.sequence_length
        hparams_dict['repeat'] = dataset.hparams.time_shift
        if args.context_length is not None:
            hparams_dict['context_frames'] = args.context_length
        if args.prediction_length is not None:
            hparams_dict['sequence_length'] = args.context_length + args.prediction_length
        return hparams_dict

    VideoPredictionModel = models.get_model_class(args.model)
    model = VideoPredictionModel(mode='test', hparams_dict=override_hparams_dict(dataset), hparams=args.model_hparams)

    if args.num_samples:
        if args.num_samples > dataset.num_examples_per_epoch():
            raise ValueError('num_samples cannot be larger than the dataset')
        num_examples_per_epoch = args.num_samples
    else:
        num_examples_per_epoch = dataset.num_examples_per_epoch()
    if num_examples_per_epoch % args.batch_size != 0:
        raise ValueError('batch_size should evenly divide the dataset')

    inputs, _ = dataset.make_batch(args.batch_size)
    if not isinstance(model, models.GroundTruthVideoPredictionModel):
        # remove ground truth data past context_frames to prevent accidentally using it
        for k, v in inputs.items():
            if k != 'actions':
                inputs[k] = v[:, :model.hparams.context_frames]

    input_phs = {k: tf.placeholder(v.dtype, v.shape, '%s_ph' % k) for k, v in inputs.items()}

    with tf.variable_scope(''):
        model.build_graph(input_phs)

    if not os.path.exists(args.output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(args.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(args), sort_keys=True, indent=4))
    with open(os.path.join(args.output_dir, "dataset_hparams.json"), "w") as f:
        f.write(json.dumps(dataset.hparams.values(), sort_keys=True, indent=4))
    with open(os.path.join(args.output_dir, "model_hparams.json"), "w") as f:
        f.write(json.dumps(model.hparams.values(), sort_keys=True, indent=4))

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_mem_frac)
    config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    sess = tf.Session(config=config)

    model.restore(sess, args.checkpoint)

    #get all tile/filenames from input_dir
    tiles = os.listdir(args.input_dir)
    tiles.sort()
    
    tile_names = []
    file_names = []
    for tile in tiles:
        in_tile_path = os.path.join(args.input_dir, tile)
        files = os.listdir(in_tile_path)
        files.sort()
        for file in files:
            tile_names.append(tile)
            file_names.append(file[:-10])
    
    #Generate and save
    sample_ind = 0
    while True:
        if args.num_samples and sample_ind >= args.num_samples:
            break
        try:
            input_results = sess.run(inputs)
        except tf.errors.OutOfRangeError:
            break
        print("Generating samples from %d to %d" % (sample_ind, sample_ind + args.batch_size))

        feed_dict = {input_ph: input_results[name] for name, input_ph in input_phs.items()}
        
        #If num stochastic samples is 1, make a mirror of input dataset with predictions
        #If num stochastic samples bigger, make a deeper subdirectory
        for stochastic_sample_ind in range(args.num_stochastic_samples):
            gen_images = sess.run(model.outputs['gen_images'], feed_dict=feed_dict)
            for i, gen_images_ in enumerate(gen_images):
                    #get only channels 0,1,2,3 send time to the last
                    gen_images_ = np.moveaxis(gen_images_[:,:,:,:4],0,-1).astype(np.float16)
                    #switch red and blue
                    r, g, b, n = np.split(gen_images_, 4, axis=2)
                    gen_images_ = np.concatenate([b,g,r,n], axis=2)
                    #Save
                    save_full_path = os.path.join(args.output_dir, tile_names[sample_ind])
                    sample_name = str(stochastic_sample_ind+1).zfill(4)+'_'+file_names[sample_ind]+'.npz'
                    if not os.path.exists(save_full_path):
                        os.makedirs(save_full_path)
                    np.savez(os.path.join(save_full_path, sample_name), gen_images_)
                    sample_ind += 1
            if stochastic_sample_ind < (args.num_stochastic_samples-1):
                sample_ind -= args.batch_size

if __name__ == '__main__':
    main()
