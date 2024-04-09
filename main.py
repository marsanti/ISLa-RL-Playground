import yaml
from utils import check_parameters
import importlib
import wandb
from algos_template.reinforce import REINFORCE
import numpy as np


def read_config(file_path):
    with open(file_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as exc:
            print(exc)

def instantiate_drl_method(params, wandb):
    method_name = params['name']
    module_name =  f"algos_template.{method_name.lower()}"  # Assuming module name is same as method name, just in lowercase
    class_name = method_name.upper()  # Assuming class name is same as method name, just in uppercase

    try:
        module = importlib.import_module(module_name)
        method_class = getattr(module, class_name)
        return method_class(params, wandb)
    except ModuleNotFoundError:
        raise ImportError(f"Module '{module_name}' not found.")
    except AttributeError:
        raise AttributeError(f"Class '{class_name}' not found in module '{module_name}'.")


if __name__ == '__main__':
    config = read_config('config.yml')

    # Accessing global Wandb settings
    use_wandb = config.get('use_wandb', False)
    wandb_config = config.get('wandb_config', None)

    print(f"Wandb Config: {wandb_config}")
    print()

    # Accessing parameters for each DRL method and env and perfom the experiment
    for method in config['DRL_methods']:

      
        # check params privided for the algorithm to evaluate
        check_parameters(method)

        # check if use manual or random seeds
        if len(method['seeds_to_test']) < 3:
            print("To few manual seeds provided, I\'ll generate three random ones to test the algorithm...")
            # test algorithm on random seeds
            seeds_to_test = [np.random.randint(0, 2000) for _ in range(3)]
        else:
           seeds_to_test = method['seeds_to_test']

       
        # training loop
        for seed in seeds_to_test:
            wandb_config['run_name'] = method['name'] + '_seed_' + str(seed)
            wandb_config['algo'] = method['name']
            wandb_config['seed'] = seed
            wandb_config['env'] = method['gym_environment']

            # instantiation of the method
            drl_method_instance = instantiate_drl_method(method, use_wandb)
            drl_method_instance.training_loop(seed, wandb_config)
            



