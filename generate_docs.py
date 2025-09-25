import inspect
import importlib

# A list of all modules that contain the components we want to document
SAMPLER_MODULES = [
    "ebm_lib.samplers.langevin",
    "ebm_lib.samplers.differentiable_langevin",
    "ebm_lib.samplers.hmc",
    "ebm_lib.samplers.parallel_tempering",
    "ebm_lib.samplers.augmented_langevin",
]

REGULARIZER_MODULES = [
    "ebm_lib.regularizers.energy",
    "ebm_lib.regularizers.gradient",
    "ebm_lib.regularizers.score_matching",
]

def generate_hyperparameter_docs(module_list, category_title):
    """
    Generates markdown documentation for a list of modules.
    """
    md_string = f"## {category_title}\n\n"

    for module_name in module_list:
        module = importlib.import_module(module_name)
        for name, obj in inspect.getmembers(module):
            # Check if it's a class defined in this module that has the static method
            if inspect.isclass(obj) and obj.__module__ == module_name and hasattr(obj, 'get_hyperparameter_info'):
                md_string += f"### `ebm_lib.{obj.__module__.split('.')[-1]}.{name}`\n\n"

                info = obj.get_hyperparameter_info()

                if not info:
                    md_string += "This module has no tunable hyperparameters.\n\n"
                    continue

                md_string += "| Hyperparameter | Type    | Description                                                   | Recommended Value(s)                               |\n"
                md_string += "|----------------|---------|---------------------------------------------------------------|----------------------------------------------------|\n"

                for param, details in info.items():
                    md_string += f"| `{param}` | `{details.get('type', 'N/A')}` | {details.get('description', '')} | {details.get('recommended', '')} |\n"

                md_string += "\n"

    return md_string

def main():
    """
    Main function to generate the full HYPERPARAMETERS.md file.
    """
    final_md = "# EBM Library Hyperparameter Guide\n\n"
    final_md += "This document provides a quick reference for the hyperparameters of the various components in this library, along with recommended starting values from the literature.\n\n"

    final_md += generate_hyperparameter_docs(SAMPLER_MODULES, "Samplers")
    final_md += generate_hyperparameter_docs(REGULARIZER_MODULES, "Regularizers")

    with open("HYPERPARAMETERS.md", "w") as f:
        f.write(final_md)

    print("Successfully generated HYPERPARAMETERS.md")

if __name__ == "__main__":
    main()