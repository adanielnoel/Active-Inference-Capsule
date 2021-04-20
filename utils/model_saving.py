import torch
import torch.distributions


def save(model, model_save_filepath):
    torch.save(model.state_dict(), model_save_filepath)


def _get_sub_model_state_dict(state_dict, sub_model_path):
    sub_model_dict = {}
    idx_next_child = len(sub_model_path)
    for key, value in state_dict.items():
        if key[:idx_next_child] == sub_model_path:
            sub_model_dict[key[idx_next_child + 1:]] = value
    return sub_model_dict


def load_capsule_parameters(model, model_save_filepath, load_vae=True, load_transition_model=True, load_biased_model=True):
    state_dict = torch.load(model_save_filepath)
    if load_vae:
        model.vae.load_state_dict(_get_sub_model_state_dict(state_dict, 'vae'))
    if load_transition_model:
        model.transition_model.load_state_dict(_get_sub_model_state_dict(state_dict, 'transition_model'))
    if load_biased_model and not isinstance(model.biased_model, torch.distributions.Distribution):
        if len([key for key in state_dict.keys() if 'biased_model.' in key]) > 0:
            model.biased_model.load_state_dict(_get_sub_model_state_dict(state_dict, 'biased_model'))

