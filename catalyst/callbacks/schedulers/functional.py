def scale_lr_for_param_groups(param_groups, initial_learning_rates, scale: float):
    if len(param_groups) != len(initial_learning_rates):
        raise ValueError()

    for original_lr, pg in zip(initial_learning_rates, param_groups):
        pg["lr"] = original_lr * scale
