import clip
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import warnings

from data import load_dataset
from utils import initialize_support
from utils import prediction_methods as clip_methods

warnings.simplefilter("ignore")

def run(args):
	model, preprocess = clip.load(args.clip_model_name, args.device)

	# if we want the descriptors in the class prompt
	if args.method == "extended_prompts":
		dataset = load_dataset(args.dataset_name, args.dataset_split, "extended", preprocess)
	else:
		dataset = load_dataset(args.dataset_name, args.dataset_split, "basic", preprocess)

	loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

	# get the prompt features
	prompt_features = torch.load(dataset.class_prompt_path).to(args.device)
	prompt_features /= prompt_features.norm(dim=-1, keepdim=True)

	# start classification
	if args.method == "base" or args.method == "extended_prompts":
		accuracy = clip_methods.base_clip_classify(args, model, loader, prompt_features)
	
	elif args.method == "support":
		descriptor_database = torch.load(args.descriptor_database_path)
		
		support = initialize_support(args.dataset_name, dataset.classes, descriptor_database)
		support_features = torch.cat(support, dim=0)
		support_features /= support_features.norm(dim=-1, keepdim=True)
		support_features = support_features.to(args.device)

		accuracy = clip_methods.support_clip_classify(args, model, loader, prompt_features, support_features)

	elif args.method == "project_prompts":
		descriptor_database = torch.load(args.descriptor_database_path)
	
		support = initialize_support(args.dataset_name, dataset.classes, descriptor_database)
		projected_prompts = clip_methods.attn_project_prompts(args, prompt_features, support)	
		
		accuracy = clip_methods.base_clip_classify(args, model, loader, projected_prompts)

	tqdm.write(f"{args.dataset_name} accuracy: {accuracy}")


if __name__ == "__main__":
	# command line args for ease
	def get_args():
		parser = argparse.ArgumentParser()
		parser.add_argument("--dataset-name", type=str, default="tinyimagenet")
		parser.add_argument("--method", type=str, default="extended_prompts", choices=["base", "support", "project_prompts", "extended_prompts"])
		parser.add_argument("--device", type=str, default="cuda")
		parser.add_argument("--clip-model-name", type=str, default="ViT-B/16")
		parser.add_argument("--batch-size", type=int, default=8)
		parser.add_argument("--dataset-split", type=str, default="val")
		parser.add_argument("--descriptor-database-path", type=str, default="../data/imagenet/imagenet_attribute_features.pt")
		args = parser.parse_args()
		return args

	args = get_args()
	run(args)

    
    
    
