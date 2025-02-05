import sys
import yaml
import importlib
from Consys import Consys

def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def instantiate_class(module_name, class_name, params):
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls(**params)

if __name__ == "__main__":
    config_path = sys.argv[1]
    config = load_config(config_path)
    
    plant = instantiate_class(
        config["plant"]["module"],
        config["plant"]["class"],
        config["plant"]["params"]
    )
    
    controller = instantiate_class(
        config["controller"]["module"],
        config["controller"]["class"],
        config["controller"]["params"]
    )
    
    sim = Consys(
        controller=controller,
        plant=plant,
        initial_state=config["simulation"]["initial_state"],
        setpoint=config["simulation"]["setpoint"],
        time_steps=config["simulation"]["time_steps"],
        disturbance_range=tuple(config["simulation"]["disturbance_range"]),
        seed=config["simulation"]["seed"]
    )
    
    optimized_gains = sim.train(
        epochs=config["training"]["epochs"],
        learning_rate=config["training"]["learning_rate"]
    )
    
    final_loss = sim.loss_fn(optimized_gains)
    print("Final loss:", final_loss)
