import os.path as op

from datasets import build_dataloader
from processor.processor import do_inference
from utils.checkpoint import Checkpointer
from utils.logger import setup_logger
from model import build_model
import argparse
from utils.iotools import load_train_configs


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="HKGR Test")
	parser.add_argument("--config_file", default='logs/CUHK-PEDES/iira/configs.yaml')
	parser.add_argument("--fold5", default=False, action='store_true')  # for coco 1K test  i2t_metric
	parser.add_argument("--i2t_metric", default=False, action='store_true')  # image to text retrieval metric
	args = parser.parse_args()
	fold5 = args.fold5

	i2t_metric = args.i2t_metric
	args = load_train_configs(args.config_file)
	args.fold5 = fold5
	args.i2t_metric = i2t_metric
	args.training = False
	logger = setup_logger('HKGR', save_dir=args.output_dir, if_train=args.training)
	logger.info(args)
	device = "cuda"

	test_img_loader, test_txt_loader, num_classes = build_dataloader(args)
	model = build_model(args, num_classes=num_classes)
	checkpointer = Checkpointer(model)
	checkpointer.load(f=op.join(args.output_dir, 'best.pth'))
	model.to(device)

	if args.dataset_name == 'coco' and args.fold5:
		do_inference(model, test_img_loader, test_txt_loader, i2t_metric=args.i2t_metric, fold5=args.fold5)
	else:
		do_inference(model, test_img_loader, test_txt_loader, i2t_metric=args.i2t_metric)