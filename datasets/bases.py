from torch.utils.data import Dataset
import logging
import torch
from utils.iotools import read_image
from utils.simple_tokenizer import SimpleTokenizer
from prettytable import PrettyTable
import random
import math
import copy


class BaseDataset(object):
	"""
    Base class of text to image reid dataset
    """
	logger = logging.getLogger("HKGR.dataset")

	def show_dataset_info(self):
		num_train_pids, num_train_imgs, num_train_captions = len(
			self.train_id_container), len(self.train_annos), len(self.train)
		num_test_pids, num_test_imgs, num_test_captions = len(
			self.test_id_container), len(self.test_annos), len(
			self.test['captions'])
		num_val_pids, num_val_imgs, num_val_captions = len(
			self.val_id_container), len(self.val_annos), len(
			self.val['captions'])

		# TODO use prettytable print comand line table

		self.logger.info(f"{self.__class__.__name__} Dataset statistics:")
		table = PrettyTable(['subset', 'ids', 'images', 'captions'])
		table.add_row(
			['train', num_train_pids, num_train_imgs, num_train_captions])
		table.add_row(
			['test', num_test_pids, num_test_imgs, num_test_captions])
		table.add_row(['val', num_val_pids, num_val_imgs, num_val_captions])
		self.logger.info('\n' + str(table))


def tokenize(caption: str, tokenizer, text_length=77, truncate=True) -> torch.LongTensor:
	sot_token = tokenizer.encoder["<|startoftext|>"]
	eot_token = tokenizer.encoder["<|endoftext|>"]
	tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]

	result = torch.zeros(text_length, dtype=torch.long)
	if len(tokens) > text_length:
		if truncate:
			tokens = tokens[:text_length]
			tokens[-1] = eot_token
		else:
			raise RuntimeError(
				f"Input {caption} is too long for context length {text_length}"
			)
	result[:len(tokens)] = torch.tensor(tokens)
	return result


class ImageTextDataset(Dataset):
	def __init__(self,
	             dataset,
	             transform=None,
	             text_length: int = 77,
	             truncate: bool = True):
		self.dataset = dataset
		self.transform = transform
		self.text_length = text_length
		self.truncate = truncate
		self.tokenizer = SimpleTokenizer()

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, index):
		pid, image_id, img_path, caption = self.dataset[index]
		img = read_image(img_path)
		if self.transform is not None:
			img = self.transform(img)

		tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

		ret = {
			'pids': pid,
			'image_ids': image_id,
			'images': img,
			'caption_ids': tokens,
		}

		return ret


class ImageDataset(Dataset):
	def __init__(self, image_pids, img_paths, transform=None):
		self.image_pids = image_pids
		self.img_paths = img_paths
		self.transform = transform

	def __len__(self):
		return len(self.image_pids)

	def __getitem__(self, index):
		pid, img_path = self.image_pids[index], self.img_paths[index]
		img = read_image(img_path)
		if self.transform is not None:
			img = self.transform(img)
		return pid, img


class TextDataset(Dataset):
	def __init__(self,
	             caption_pids,
	             captions,
	             text_length: int = 77,
	             truncate: bool = True):
		self.caption_pids = caption_pids
		self.captions = captions
		self.text_length = text_length
		self.truncate = truncate
		self.tokenizer = SimpleTokenizer()

	def __len__(self):
		return len(self.caption_pids)

	def __getitem__(self, index):
		pid, caption = self.caption_pids[index], self.captions[index]

		caption = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

		return pid, caption


class ImageTextMLMDataset(Dataset):
	def __init__(self,
	             dataset, parsed_cap,
	             transform=None,
	             text_length: int = 77,
	             mlm_type=None,
	             truncate: bool = True):
		self.dataset = dataset
		self.transform = transform
		self.text_length = text_length
		self.truncate = truncate
		self.parsed_cap = parsed_cap

		self.tokenizer = SimpleTokenizer()
		self.vocab = list(self.tokenizer.encoder.keys())[1:-3]
		self.mask = self.tokenizer.encoder["<|mask|>"]
		self.mlm_type = [l.strip() for l in mlm_type.split('+')]

		self.mask_func = {0: self._build_random_masked_tokens_and_labels, 1: self._build_object_masked_tokens_and_labels, 
						  2: self._build_attribute_masked_tokens_and_labels, 3: self._build_relation_masked_tokens_and_labels}
		self.mask_type = {0: "rand", 1: "object", 2: "attribute", 3: "relation"}

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, index):
		pid, image_id, img_path, caption = self.dataset[index]
		parsed_caption = self.parsed_cap[index]
		img = read_image(img_path)
		if self.transform is not None:
			img = self.transform(img)

		caption_tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length,
		                          truncate=self.truncate)

		mlm_caption_tokens = copy.deepcopy(caption_tokens.cpu().numpy())

		if len(self.mlm_type) == 1 and 'rand' in self.mlm_type:
			mlm_tokens, mlm_labels = self._build_random_masked_tokens_and_labels(mlm_caption_tokens)
			ret = {
				'pids': pid,
				'image_ids': image_id,
				'images': img,
				'caption_ids': caption_tokens,
				'mlm_ids': mlm_tokens,
				'mlm_labels': mlm_labels
			}
			return ret

		elif len(self.mlm_type) == 1 and 'obj' in self.mlm_type:
			mlm_tokens_obj, mlm_labels_obj = self._build_object_masked_tokens_and_labels(caption, mlm_caption_tokens, parsed_caption['object'])
			ret = {
				'pids': pid,
				'image_ids': image_id,
				'images': img,
				'caption_ids': caption_tokens,
				'mlm_ids': mlm_tokens_obj,
				'mlm_labels': mlm_labels_obj
			}
			return ret

		elif len(self.mlm_type) == 1 and 'attr' in self.mlm_type:
			mlm_tokens_attr, mlm_labels_attr = self._build_attribute_masked_tokens_and_labels(caption, mlm_caption_tokens, parsed_caption['attribute'])
			ret = {
				'pids': pid,
				'image_ids': image_id,
				'images': img,
				'caption_ids': caption_tokens,
				'mlm_ids': mlm_tokens_attr,
				'mlm_labels': mlm_labels_attr
			}
			return ret

		elif len(self.mlm_type) == 1 and 'rel' in self.mlm_type:
			mlm_tokens_rel, mlm_labels_rel = self._build_relation_masked_tokens_and_labels(caption, mlm_caption_tokens, parsed_caption['relation'])
			ret = {
				'pids': pid,
				'image_ids': image_id,
				'images': img,
				'caption_ids': caption_tokens,
				'mlm_ids': mlm_tokens_rel,
				'mlm_labels': mlm_labels_rel
			}
			return ret

		else:
			rand_num = random.randint(0,3)
			mask_type = self.mask_type[rand_num]
			print(mask_type)
			if mask_type == "rand":
				mlm_tokens, mlm_labels = self.mask_func[rand_num](mlm_caption_tokens)
			else:
				mlm_tokens, mlm_labels = self.mask_func[rand_num](caption, mlm_caption_tokens, parsed_caption[mask_type])

			ret = {
				'pids': pid,
				'image_ids': image_id,
				'images': img,
				'caption_ids': caption_tokens,
				'mlm_ids': mlm_tokens,
				'mlm_labels': mlm_labels
			}

			return ret

	def _build_random_masked_tokens_and_labels(self, mlm_caption_tokens):
		"""
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of int, tokenized sentence.
        :return: (list of int, list of int), masked tokens and related labels for MLM prediction
        masked tokens: <|startoftext|>a atures with medium length brown hair is wearing a large white shirt and tight blue jeans <|mask|><|endoftext|>
        """
		mask = self.tokenizer.encoder["<|mask|>"]
		token_range = list(range(1, len(self.tokenizer.encoder) - 3))  # 1 ~ 49405 最后3个: mask, start, end

		labels = []
		for i, token in enumerate(mlm_caption_tokens):
			if 0 < token < 49405:  # <|start|>: 49406, <|end|>: 49407
				prob = random.random()
				# mask token with 15% probability
				if prob < 0.15:
					prob /= 0.15

					# 80% randomly change token to mask token
					if prob < 0.8:
						mlm_caption_tokens[i] = mask

					# 10% randomly change token to random token
					elif prob < 0.9:
						mlm_caption_tokens[i] = random.choice(token_range)

					# -> rest 10% randomly keep current token

					# append current token to output (we will predict these later)
					labels.append(token)
				else:
					# no masking token (will be ignored by loss function later)
					labels.append(0)
			else:
				labels.append(0)

		if all(l == 0 for l in labels):
			# at least mask 1
			labels[1] = mlm_caption_tokens[1]
			mlm_caption_tokens[1] = mask

		return torch.tensor(mlm_caption_tokens), torch.tensor(labels)

	def _build_object_masked_tokens_and_labels(self, caption, mlm_caption_tokens, object_words):
		"""
		Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
		:param tokens: list of int, tokenized sentence.
		:return: (list of int, list of int), masked tokens and related labels for MLM prediction
		masked tokens: <|startoftext|>a atures with medium length brown hair is wearing a large white shirt and tight blue jeans <|mask|><|endoftext|>
		"""
		chosed_num = math.ceil(len(object_words) * 0.3)
		if chosed_num == 0:
			chosed_objects = random.sample(caption.split(" "), 1)
		else:
			chosed_objects = random.sample(object_words, int(chosed_num))  # randomly select 30% of them to mask

		objects_tokens = [self.tokenizer.encode(i) for i in chosed_objects]

		mask = self.tokenizer.encoder["<|mask|>"]
		token_range = list(range(1, len(self.tokenizer.encoder) - 3))  # 1 ~ 49405 最后3个: mask, start, end

		labels = [0] * len(mlm_caption_tokens)
		for obj_token in objects_tokens:
			prob = random.random()
			for i, token in enumerate(mlm_caption_tokens):
				if 0 < token < 49405:  # <|start|>: 49406, <|end|>: 49407
					if token in obj_token:  # t-shirt: [339, 268, 2523]
						for sub_token in obj_token:
							if token == sub_token:
								if prob < 0.8:  # 80% change token to mask token
									mlm_caption_tokens[i] = mask
									labels[i] = token
								# 10% randomly change token to random token
								elif 0.8 <= prob < 0.9:
									mlm_caption_tokens[i] = random.choice(token_range)
									# append current token to output (we will predict these later)
									labels[i] = token
								else:
									# -> rest 10% randomly keep current token
									# no masking token (will be ignored by loss function later)
									labels[i] = 0

		if all(l == 0 for l in labels):
			# at least mask 1
			labels[1] = mlm_caption_tokens[1]
			mlm_caption_tokens[1] = mask


		return torch.tensor(mlm_caption_tokens), torch.tensor(labels)

	def _build_attribute_masked_tokens_and_labels(self, caption, mlm_caption_tokens, attribute_words):
		"""
		Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
		:param tokens: list of int, tokenized sentence.
		:return: (list of int, list of int), masked tokens and related labels for MLM prediction
		masked tokens: <|startoftext|>a atures with medium length brown hair is wearing a large white shirt and tight blue jeans <|mask|><|endoftext|>
		"""
		# attribute_words = list(set(attribute_words))
		chosed_num = math.ceil(len(attribute_words) * 0.3)
		if chosed_num == 0:
			chosed_attribute = random.sample(caption.split(" "), 1)
		else:
			chosed_attribute = random.sample(attribute_words, int(chosed_num))  # randomly select 30% of them to mask

		chosed_attribute = list(set(chosed_attribute))
		attribute_tokens = [self.tokenizer.encode(i) for i in chosed_attribute]

		mask = self.tokenizer.encoder["<|mask|>"]
		token_range = list(range(1, len(self.tokenizer.encoder) - 3))  # 1 ~ 49405 最后3个: mask, start, end

		labels = [0] * len(mlm_caption_tokens)
		for attr_token in attribute_tokens:
			prob = random.random()
			for i, token in enumerate(mlm_caption_tokens):
				if 0 < token < 49405:  # <|start|>: 49406, <|end|>: 49407
					if token in attr_token:  # t-shirt: [339, 268, 2523]
						for sub_token in attr_token:
							if token == sub_token:
								if prob < 0.8:  # 80% change token to mask token
									mlm_caption_tokens[i] = mask
									labels[i] = token
								# 10% randomly change token to random token
								elif 0.8 <= prob < 0.9:
									mlm_caption_tokens[i] = random.choice(token_range)
									# append current token to output (we will predict these later)
									labels[i] = token
								else:
									# -> rest 10% randomly keep current token
									# no masking token (will be ignored by loss function later)
									labels[i] = 0

		if all(l == 0 for l in labels):
			# at least mask 1
			labels[1] = mlm_caption_tokens[1]
			mlm_caption_tokens[1] = mask

		return torch.tensor(mlm_caption_tokens), torch.tensor(labels)

	def _build_relation_masked_tokens_and_labels(self, caption, mlm_caption_tokens, relation_words):
		"""
		Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
		:param tokens: list of int, tokenized sentence.
		:return: (list of int, list of int), masked tokens and related labels for MLM prediction
		masked tokens: <|startoftext|>a atures with medium length brown hair is wearing a large white shirt and tight blue jeans <|mask|><|endoftext|>
		"""
		# attribute_words = list(set(attribute_words))
		chosed_num = math.ceil(len(relation_words) * 0.3)
		if chosed_num == 0:
			chosed_relation = random.sample(caption.split(" "), 1)
		else:
			chosed_relation = random.sample(relation_words, int(chosed_num))  # randomly select 30% of them to mask

		chosed_relation = list(set(chosed_relation))
		relation_tokens = [self.tokenizer.encode(i) for i in chosed_relation]

		mask = self.tokenizer.encoder["<|mask|>"]
		token_range = list(range(1, len(self.tokenizer.encoder) - 3))  # 1 ~ 49405 最后3个: mask, start, end

		labels = [0] * len(mlm_caption_tokens)
		for rel_token in relation_tokens:
			prob = random.random()
			for i, token in enumerate(mlm_caption_tokens):
				if 0 < token < 49405:  # <|start|>: 49406, <|end|>: 49407
					if token in rel_token:  # t-shirt: [339, 268, 2523]
						for sub_token in rel_token:
							if token == sub_token:
								if prob < 0.8:  # 80% change token to mask token
									mlm_caption_tokens[i] = mask
									labels[i] = token
								# 10% randomly change token to random token
								elif 0.8 <= prob < 0.9:
									mlm_caption_tokens[i] = random.choice(token_range)
									# append current token to output (we will predict these later)
									labels[i] = token
								else:
									# -> rest 10% randomly keep current token
									# no masking token (will be ignored by loss function later)
									labels[i] = 0

		if all(l == 0 for l in labels):
			# at least mask 1
			labels[1] = mlm_caption_tokens[1]
			mlm_caption_tokens[1] = mask

		return torch.tensor(mlm_caption_tokens), torch.tensor(labels)
