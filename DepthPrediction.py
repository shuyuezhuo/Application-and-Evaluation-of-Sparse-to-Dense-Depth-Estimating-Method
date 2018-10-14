import os
import time
import cv2
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from pykinect2 import PyKinectRuntime
from pykinect2 import PyKinectV2

import utils
from metrics import Result
import dataloaders.transforms as transforms

args = utils.parse_command()


def main():
    print('Testing data on ' + args.camera + '!')

    assert args.data == 'nyudepthv2', '=> only nyudepthv2 ' \
                                      'available at this ' \
                                      'point'

    to_tensor = transforms.ToTensor()

    assert not (
                args.camera == 'webcam' and not
    args.modality == 'rgb'), '=> webcam only accept RGB ' \
                             'model'

    output_directory = utils.get_output_directory(args)
    best_model_filename = os.path.join(output_directory,
                                       'model_best.pth.tar')
    assert os.path.isfile(best_model_filename), \
        "=> no best model found at '{}'".format(
            best_model_filename)
    print("=> loading best model '{}'".format(
        best_model_filename))
    checkpoint = torch.load(best_model_filename)
    args.start_epoch = checkpoint['epoch']
    model = checkpoint['model']
    model.eval()

    switch = True

    if args.camera == 'kinect':
        kinect = PyKinectRuntime.PyKinectRuntime(
            PyKinectV2.FrameSourceTypes_Color |
            PyKinectV2.FrameSourceTypes_Depth)
        counter = 0

        assert not kinect._sensor is None, '=> No Kinect ' \
                                           'device ' \
                                           'detected!'

        while True:
            if kinect.has_new_color_frame() and \
                    kinect.has_new_depth_frame():
                bgra_frame = kinect.get_last_color_frame()
                bgra_frame = bgra_frame.reshape((
                                                kinect.color_frame_desc.Height,
                                                kinect.color_frame_desc.Width,
                                                4),
                                                order='C')
                rgb_frame = cv2.cvtColor(bgra_frame,
                                         cv2.COLOR_BGRA2RGB)

                depth_frame = kinect.get_last_depth_frame()

                merged_image, rmse = depth_estimate(model,
                                                    rgb_frame,
                                                    depth_frame,
                                                    save=False)

                merged_image_bgr = cv2.cvtColor(
                    merged_image.astype('uint8'),
                    cv2.COLOR_RGB2BGR, switch)
                switch = False
                cv2.imshow('my webcam',
                           merged_image_bgr.astype('uint8'))
                if counter == 15:
                    print('RMSE = ' + str(rmse))
                counter = counter + 1
                if counter == 16:
                    counter = 0

            if cv2.waitKey(1) == 27:
                break

    elif args.camera == 'webcam':

        cam = cv2.VideoCapture(0)

        while True:

            ret_val, img = cam.read()

            img = cv2.flip(img, 1)

            rgb = cv2.cvtColor(np.array(img),
                               cv2.COLOR_BGRA2RGB)

            transform = transforms.Compose([
                transforms.Resize([228, 304]),
            ])
            rgb_image = transform(rgb)

            if args.modality == 'rgbd':
                assert '=> can\'t test webcam with depth ' \
                       'information!'

            rgb_np = np.asfarray(rgb_image,
                                 dtype='float') / 255
            input_tensor = to_tensor(rgb_np)
            while input_tensor.dim() < 4:
                input_tensor = input_tensor.unsqueeze(0)
            input_tensor = input_tensor.cuda()
            torch.cuda.synchronize()
            end = time.time()
            with torch.no_grad():
                pred = model(input_tensor)
            torch.cuda.synchronize()
            gpu_time = time.time() - end

            pred_depth = np.squeeze(pred.cpu().numpy())

            d_min = np.min(pred_depth)
            d_max = np.max(pred_depth)
            pred_color_map = color_map(pred_depth, d_min,
                                       d_max,
                                       plt.cm.viridis)

            merged_image = np.hstack(
                [rgb_image, pred_color_map])
            merged_image_bgr = cv2.cvtColor(
                merged_image.astype('uint8'),
                cv2.COLOR_RGB2BGR)
            cv2.imshow('my webcam',
                       merged_image_bgr.astype('uint8'))

            if cv2.waitKey(1) == 27:
                break  # esc to quit

    else:
        file_name = args.kinectdata + '.p'
        pickle_path = os.path.join('CameraData', file_name)
        print(pickle_path)

        if not os.path.exists('CameraData'):
            assert '=>do data find at ' + pickle_path

        f = open(pickle_path, 'rb')
        pickle_file = pickle.load(f)
        f.close()

        bgr_frame = pickle_file['rgb']
        depth = pickle_file['depth']

        rgb_frame = cv2.cvtColor(bgr_frame,
                                 cv2.COLOR_BGR2RGB)

        merged_image, rmse = depth_estimate(model,
                                            rgb_frame,
                                            depth,
                                            save=True,
                                            switch=True)
        plt.figure('Merged Image')
        plt.imshow(merged_image.astype('uint8'))
        plt.show()
        print('RMSE = ' + str(rmse))

    cv2.destroyAllWindows()


def depth_estimate(model, rgb_frame, depth, save=False,
                   switch=False):
    to_tensor = transforms.ToTensor()
    cmap_depth = plt.cm.viridis
    cmap_error = plt.cm.inferno

    rgb_np, depth_np = image_transform(rgb_frame, depth)

    ##creat sparse depth
    mask_keep = sampler(depth_np, args.sample_spacing)
    sample_number = np.count_nonzero(
        mask_keep.astype('int'))
    if args.modality == 'rgb':
        sample_number = 0
    if switch:
        print('Total samples = ' + str(sample_number))
    sparse_depth = np.zeros(depth_np.shape)
    sparse_depth[mask_keep] = depth_np[mask_keep]
    sparse_depth_np = sparse_depth

    ##choose input
    if args.modality == 'rgb':
        input_np = rgb_np
    elif args.modality == 'rgbd':
        rgbd = np.append(rgb_np,
                         np.expand_dims(sparse_depth_np,
                                        axis=2), axis=2)
        input_np = np.asfarray(rgbd, dtype='float')
    elif args.modality == 'd':
        input_np = sparse_depth_np

    input_tensor = to_tensor(input_np)

    while input_tensor.dim() < 4:
        input_tensor = input_tensor.unsqueeze(0)

    input_tensor = input_tensor.cuda()
    torch.cuda.synchronize()

    ##get prediction
    end = time.time()
    with torch.no_grad():
        pred = model(input_tensor)
    torch.cuda.synchronize()
    gpu_time = time.time() - end

    ##get result
    target_tensor = to_tensor(depth_np)
    while target_tensor.dim() < 4:
        target_tensor = target_tensor.unsqueeze(0)
    target_tensor = target_tensor.cuda()
    torch.cuda.synchronize()
    result = Result()
    result.evaluate(pred.data, target_tensor.data)

    pred_depth = np.squeeze(pred.data.cpu().numpy())

    ##convert depth to colour map
    d_min = min(np.min(pred_depth), np.min(depth_np))
    d_max = max(np.max(pred_depth), np.max(depth_np))
    # # d_min = float(0.5)
    # # d_max = float(6)

    pred_depth_rgb_map = color_map(pred_depth, d_min, d_max,
                                   cmap_depth)
    input_depth_color_map = color_map(depth_np, d_min,
                                      d_max, cmap_depth)
    sparse_depth_color_map = color_map(sparse_depth, d_min,
                                       d_max, cmap_depth)

    ##get error map
    mask = depth_np <= 0
    combined_map = depth_np
    combined_map[mask] = pred_depth[mask]
    abs_error_map = np.absolute(combined_map - pred_depth)
    # error_min = min(np.min(combined_map), np.min(
    # pred_depth))
    # error_max = max(np.max(combined_map), np.max(
    # pred_depth))
    error_min = np.min(abs_error_map)
    error_max = np.max(abs_error_map)
    error_map_color = color_map(abs_error_map, error_min,
                                error_max, cmap_error)

    ##show images
    rgb_image = rgb_np * 255
    merged_image = np.hstack([rgb_image, pred_depth_rgb_map,
                              input_depth_color_map,
                              sparse_depth_color_map,
                              error_map_color])

    ##save image
    if args.write and save:
        images_save(sample_number, pred_depth_rgb_map,
                    sparse_depth_color_map, error_map_color,
                    rgb_image, input_depth_color_map,
                    result)

    return merged_image, result.rmse


################################################################
def sampler(depth, sample_spcaing):
    mask = np.zeros(depth.shape, dtype=bool)
    for height in np.arange(5, depth.shape[0] - 5):
        for width in np.arange(5, depth.shape[1] - 5):
            if width % sample_spcaing == 0 and height % \
                    sample_spcaing == 0:
                mask[height, width] = True
    return mask


##################################################################
def images_save(sample_number, pred_depth_rgb_map,
                sparse_depth_color_map, error_map_color,
                rgb_image, input_depth_color_map, result):
    image_directory = os.path.join('image_result',
                                   args.kinectdata)
    if not os.path.exists(image_directory):
        os.makedirs(image_directory)

    ##save depth prediction
    depth_pre_image_directory = os.path.join(
        image_directory,
        "scene={}.input_sample={}.train_samples={"
        "}.modality={}.arch={}.decoder={}.depth.png". \
        format(args.kinectdata, sample_number,
               args.num_samples, args.modality, \
               args.arch, args.decoder))
    img.imsave(depth_pre_image_directory,
               pred_depth_rgb_map.astype('uint8'))

    ##save sparse depth
    if not args.modality == 'rgb':
        depth_sparse_image_directory = os.path.join(
            image_directory,
            "scene={}.input_sample={}.train_samples={"
            "}.modality={}.arch={}.decoder={"
            "}.sparse_depth.png". \
            format(args.kinectdata, sample_number,
                   args.num_samples,
                   args.modality, \
                   args.arch, args.decoder))
        img.imsave(depth_sparse_image_directory,
                   sparse_depth_color_map.astype('uint8'))

    ##save error heat map
    error_map_image_directory = os.path.join(
        image_directory,
        "scene={}.input_sample={}.train_samples={"
        "}.modality={}.arch={}.decoder={}.error_map.png". \
        format(args.kinectdata, sample_number,
               args.num_samples,
               args.modality, \
               args.arch, args.decoder))
    img.imsave(error_map_image_directory,
               error_map_color.astype('uint8'))

    ##save rgb and depth input
    img.imsave(os.path.join(image_directory, 'rgb.png'),
               rgb_image.astype('uint8'))
    img.imsave(os.path.join(image_directory, 'depth.png'),
               input_depth_color_map.astype('uint8'))

    ##write text result
    text_directory = os.path.join(image_directory,
                                  "scene={"
                                  "}.input_sample={"
                                  "}.train_samples={"
                                  "}.modality={}.arch={"
                                  "}.decoder={"
                                  "}.result.txt". \
                                  format(args.kinectdata,
                                         sample_number,
                                         args.num_samples,
                                         args.modality, \
                                         args.arch,
                                         args.decoder))
    with open(text_directory, 'w') as txtfile:
        txtfile.write(
            "mse={:.3f}\nrmse={:.3f}\nabsrel={:.3f}\nlg10={:.3f}\nmae={:.3f}\ndelta1={:.3f}\nt_gpu={:.4f}\n".
                format(result.mse, result.rmse,
                       result.absrel, result.lg10,
                       result.mae, result.delta1,
                       result.gpu_time))


#######################################################################
def color_map(depth, d_min=None, d_max=None, cmap=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:, :, :3]  # H, W, C


#########################################################################
def image_transform(rgb, depth):
    depth_frame_converted = np.asfarray(depth.clip(0, 6000),
                                        dtype='float') / 1000
    depth_array = depth_frame_converted.reshape((424, 512),
                                                order='C')

    rgb_transform = transforms.Compose([
        transforms.Resize([240, 426]),
        transforms.CenterCrop((228, 304)),
    ])
    depth_transform = transforms.Compose([
        transforms.Resize([240, 320]),
        transforms.CenterCrop((228, 304)),
    ])
    rgb_frame = rgb_transform(rgb)
    rgb_np = np.asfarray(rgb_frame, dtype='float') / 255
    depth_np = depth_transform(depth_array)

    return rgb_np, depth_np


if __name__ == '__main__':
    main()
