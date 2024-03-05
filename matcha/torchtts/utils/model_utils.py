def model_summary_string(model, idx=0):
    from torch.nn.modules.module import _addindent

    def _name_params(m):
        parameters = filter(lambda p: p.requires_grad, m.parameters())
        parameters = sum([p.numel() for p in parameters]) / 1_000_000
        return f"{parameters:.4f}M"

    # We treat the extra repr like the sub-module, one item per line
    extra_lines = []
    extra_repr = model.extra_repr()
    # empty string will be split into list ['']
    if extra_repr:
        extra_lines = extra_repr.split("\n")
    child_lines = []
    for key, module in model._modules.items():
        mod_str = model_summary_string(module, idx + 1)
        mod_str = _addindent(mod_str, 4)
        child_lines.append("(" + key + ": " + _name_params(module) + "): " + mod_str)
    lines = extra_lines + child_lines

    if idx == 0:
        main_str = model._get_name() + "(" + _name_params(model) + ")" + "("
    else:
        main_str = model._get_name() + "("
    if lines:
        # simple one-liner info, which most builtin Modules will use
        if len(extra_lines) == 1 and not child_lines:
            main_str += extra_lines[0]
        else:
            main_str += "\n  " + "\n  ".join(lines) + "\n"

    main_str += ")"
    return main_str
