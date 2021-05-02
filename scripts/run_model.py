#!/usr/bin/python

import argparse, json, os

from imageio import imwrite
import torch

from sg2im.model import Sg2ImModel
from sg2im.data.utils import imagenet_deprocess_batch
import sg2im.vis as vis


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='sg2im-models/vg128.pt')
parser.add_argument('--scene_graphs_json', default='scene_graphs/figure_6_sheep.json')
parser.add_argument('--output_dir', default='outputs')
parser.add_argument('--draw_scene_graphs', type=int, default=0)
parser.add_argument('--device', default='gpu', choices=['cpu', 'gpu'])


def main(args):
  if not os.path.isfile(args.checkpoint):
    print("没找到模型")
    return

  if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)

  if args.device == 'cpu':
    device = torch.device('cpu')
  elif args.device == 'gpu':
    device = torch.device('cuda:0')
    if not torch.cuda.is_available():
      device = torch.device('cpu')

  map_location = 'cpu' if device == torch.device('cpu') else None
  checkpoint = torch.load(args.checkpoint, map_location=map_location)
  model = Sg2ImModel(**checkpoint['model_kwargs'])
  model.load_state_dict(checkpoint['model_state'])
  model.eval()
  model.to(device)

  # 加载Json文本
  with open(args.scene_graphs_json, 'r') as f:
    scene_graphs = json.load(f)

  # 载入模型
  with torch.no_grad():
    imgs, boxes_pred, masks_pred, _ = model.forward_json(scene_graphs)
  imgs = imagenet_deprocess_batch(imgs)

  # 保存生成的图片
  for i in range(imgs.shape[0]):
    img_np = imgs[i].numpy().transpose(1, 2, 0)
    img_path = os.path.join(args.output_dir, 'img%06d.png' % i)
    imwrite(img_path, img_np)

  if args.draw_scene_graphs == 1:
    for i, sg in enumerate(scene_graphs):
      sg_img = vis.draw_scene_graph(sg['objects'], sg['relationships'])
      sg_img_path = os.path.join(args.output_dir, 'sg%06d.png' % i)
      imwrite(sg_img_path, sg_img)


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)

