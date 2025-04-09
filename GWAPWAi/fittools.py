#!/usr/bin/env python3
import numpy as np
import torch

def pack_parameters(observ_dict, wave="P"):
    """
    Packs model parameters into a flat vector for optimization, and provides
    an unpacking function to restore them.

    Returns:
        flat_params (tensor): 1D tensor of all parameters
        unpack_fn (function): callable to get original tensors back
        static_params (dict): non-learnable config values
    """
    wave_data = observ_dict["wave_data"][wave]
    chebys = wave_data["ChebyCoeffs"]["coeffs"]
    pole_couplings = wave_data["Poles"]["couplings"]
    pole_masses = wave_data["Poles"]["mass"]
    kbkgrd = wave_data["KmatBackground"]
    masses = observ_dict["channels"]["masses"]
    num_ch = len(observ_dict["channels"]["names"])

    # Collect shapes and flat tensors
    tensors = [chebys, pole_couplings, pole_masses, kbkgrd]
    shapes = [t.shape for t in tensors]
    sizes = [t.numel() for t in tensors]
    flat_parts = [t.reshape(-1) for t in tensors]

    # Combine everything
    flat_params = torch.cat(flat_parts)

    def unpack_fn(flat_vector):
        """Unpacks the flat vector into the original tensors."""
        idx = 0
        restored = []
        for shape, size in zip(shapes, sizes):
            chunk = flat_vector[idx:idx+size].reshape(shape)
            restored.append(chunk)
            idx += size
        return restored  # order: chebys, pole_couplings, pole_masses, kbkgrd

    static_params = {
        "J": wave_data["J"],
        "alpha": 1,
        "sL": wave_data["sL"],
        "s0": 1,
        "smin": observ_dict["fitting"]["FitRegion"][0],
        "smax": observ_dict["fitting"]["FitRegion"][1],
        "ktype": wave_data["kmat_type"],
        "rhoNtype": wave_data["rhoN_type"],
        "masses": masses,
        "num_ch": num_ch
    }

    return flat_params, unpack_fn, static_params
