#!/usr/bin/env python3
import numpy as np
import torch
from torch import nn

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

class ChiSquaredLossModule(nn.Module):
    def __init__(
        self,
        initial_params,
        unpack_fn,
        static_params,
        s_vals_c,
        II,
        values,
        errors,
        masks,
        model_channel_indices,
        weight_vector,
        compute_intensity_func,
        numerator_func,
        omega_pole_func,
        construct_phsp_func,
        K_nominal_func,
        momentum_func
    ):
        super().__init__()
        self.params = nn.Parameter(initial_params.clone().detach().to(dtype=torch.float64))
        self.unpack_fn = unpack_fn
        self.static_params = static_params
        self.s_vals_c = s_vals_c
        self.II = II
        self.values = values
        self.errors = errors
        self.masks = masks
        self.model_channel_indices = model_channel_indices
        self.weight_vector = weight_vector
        self.compute_intensity_func = compute_intensity_func
        self.numerator_func = numerator_func
        self.omega_pole_func = omega_pole_func
        self.construct_phsp_func = construct_phsp_func
        self.K_nominal_func = K_nominal_func
        self.momentum_func = momentum_func

    def forward(self):
        # Step 1: Compute model intensity
        intensity = self.compute_intensity_func(
            s=self.s_vals_c,
            flat_params=self.params,
            unpack_fn=self.unpack_fn,
            static_params=self.static_params,
            II=self.II,
            numerator_func=self.numerator_func,
            omega_pole_func=self.omega_pole_func,
            construct_phsp_func=self.construct_phsp_func,
            K_nominal_func=self.K_nominal_func,
            momentum_func=self.momentum_func
        ).real  # shape: [N_pts, N_model_channels]

        # Step 2: Build model predictions aligned with data shape
        N_pts, N_data_ch = self.values.shape
        model_grid = torch.zeros((N_pts, N_data_ch), device=self.params.device, dtype=self.params.dtype)

        # Broadcast from model channels into the data channels
        is_inclusive = self.model_channel_indices == -1
        is_regular = ~is_inclusive

        # Regular channels: direct column copy
        model_grid[:, is_regular] = intensity[:, self.model_channel_indices[is_regular]]

        # Inclusive: sum over all model channels
        inclusive_pred = intensity.sum(dim=1, keepdim=True)  # [N_pts, 1]
        model_grid[:, is_inclusive] = inclusive_pred

        # Step 3: Mask-safe division and residuals
        safe_errors = torch.where(self.masks, self.errors, torch.ones_like(self.errors))
        residuals = (model_grid - self.values) / safe_errors
        residuals = residuals * self.masks  # only count real data points

        # Step 4: Apply per-channel weights and return chi²
        weighted_residuals = residuals * self.weight_vector.view(1, -1)
        return torch.sum(weighted_residuals ** 2)



def run_fit(
    model,
    initial_params,
    num_starts=50,
    adam_iters=100,
    lbfgs_iters=300,
    adam_lr=1e-2
):
    best_loss = float('inf')
    best_params = None

    for start in range(num_starts):
        print(f"\n=== Start {start + 1}/{num_starts} ===")

        # Randomize starting point near initial
        noise = torch.randn_like(initial_params) * 0.1
        #print(noise.device)
        model.params.data = initial_params.clone().detach() + noise
        #print(model.params.data)

        # === Adam: global search ===
        optimizer = torch.optim.Adam([model.params], lr=adam_lr)
        for i in range(adam_iters):
            optimizer.zero_grad()
            loss = model()
            if torch.isnan(loss):
                print("NaN encountered during Adam")
                break
            loss.backward()
            optimizer.step()
            if i % 50 == 0:
                grads = torch.cat([p.grad.flatten() for p in model.parameters()])
                print(f"[Adam iter {i:3d}] χ² = {loss.item():.4f}, max |grad| = {grads.abs().max().item():.4e}")

        # === LBFGS: refine minimum ===
        optimizer = torch.optim.LBFGS([model.params], max_iter=lbfgs_iters, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            loss = model()
            if torch.isnan(loss):
                raise RuntimeError("NaN during LBFGS closure")
            loss.backward()
            return loss

        try:
            optimizer.step(closure)
        except RuntimeError as e:
            print("LBFGS error:", e)
            continue

        final_loss = model().item()
        print(f"Final χ² after LBFGS: {final_loss:.4f}")

        if final_loss < best_loss:
            best_loss = final_loss
            best_params = model.params.detach().clone()

    # Set model to best solution
    if best_params is not None:
        model.params.data = best_params
        print("\nBest-fit χ²:", best_loss)
    else:
        raise RuntimeError("All fits failed.")

    return best_params


def randomize_initial_params(observ_dict, unpack_fn, wave="P"):
    """
    Generate a starting point for flat_params using the dictionary values
    and their associated uncertainties.

    Parameters:
        observ_dict : dict
            The original parameter dictionary.
        unpack_fn : function
            The unpacking function returned by pack_parameters.
        wave : str
            The partial wave to extract values for.

    Returns:
        flat_params : torch.Tensor
    """
    wave_data = observ_dict["wave_data"][wave]

    # Chebyshev coefficients
    chebys = wave_data["ChebyCoeffs"]["coeffs"].clone()
    cheby_errs = wave_data["ChebyCoeffs"].get("errors", torch.ones_like(chebys) * 0.1)
    chebys = chebys + cheby_errs * torch.randn_like(chebys)

    # Pole masses and couplings
    poles = wave_data["Poles"]["mass"].clone()
    pole_errs = wave_data["Poles"].get("mass_err", torch.ones_like(poles) * 0.1)
    poles = poles + pole_errs * torch.randn_like(poles)

    couplings = wave_data["Poles"]["couplings"].clone()
    coupling_errs = wave_data["Poles"].get("coupling_err", torch.ones_like(couplings) * 0.1)
    couplings = couplings + coupling_errs * torch.randn_like(couplings)

    # K-matrix background
    kbkgrd = wave_data["KmatBackground"].clone()
    noise = 0.05 * torch.randn_like(kbkgrd)  # Optional light noise
    kbkgrd = kbkgrd + noise

    # Pack back into flat_params
    return torch.cat([
        chebys.flatten(),
        couplings.flatten(),
        poles.flatten(),
        kbkgrd.flatten()
    ])


def plot_fit_vs_data_full(
    best_fit_params,
    model,
    sqrt_s_grid,
    values,
    errors,
    masks,
    channel_list
):
    import matplotlib.pyplot as plt
    import numpy as np

    # Compute model intensity
    intensity = model.compute_intensity_func(
        s=sqrt_s_grid**2,
        flat_params=best_fit_params,
        unpack_fn=model.unpack_fn,
        static_params=model.static_params,
        II=model.II,
        numerator_func=model.numerator_func,
        omega_pole_func=model.omega_pole_func,
        construct_phsp_func=model.construct_phsp_func,
        K_nominal_func=model.K_nominal_func,
        momentum_func=model.momentum_func
    ).real
    
    s_np = (sqrt_s_grid).detach().cpu().numpy().real
    intensity_np = intensity.detach().cpu().numpy()
    values_np = values.detach().cpu().numpy()
    errors_np = errors.detach().cpu().numpy()
    masks_np = masks.detach().cpu().numpy()

    all_channels = model.static_params["channel_names"]
    num_channels = len(all_channels)
    has_inclusive = "inclusive" in channel_list

    total_plots = num_channels + (1 if has_inclusive else 0)

    fig, axes = plt.subplots(total_plots, 2, figsize=(12, 4 * total_plots), gridspec_kw={"width_ratios": [3, 2]})
    fig.suptitle("Best-Fit Intensity and Residuals vs Experimental Data", fontsize=16)

    chi2_total = 0.0

    for plot_idx in range(num_channels):
        label = all_channels[plot_idx]
        ax_intensity = axes[plot_idx, 0]
        ax_residuals = axes[plot_idx, 1]

        ax_intensity.plot(s_np, intensity_np[:, plot_idx], label="Model", color="tab:blue")
        ax_intensity.set_title(f"{label} Intensity")
        ax_intensity.set_ylabel("Intensity")
        ax_intensity.grid(True)
        ax_intensity.legend()

        if label in channel_list:
            j = channel_list.index(label)
            mask = masks_np[:, j]
            s_data = s_np[mask]
            y_data = values_np[mask, j]
            y_err = errors_np[mask, j]
            y_model = intensity_np[mask, plot_idx]

            residuals = (y_model - y_data) / y_err
            chi2 = np.sum(residuals**2)
            chi2_total += chi2

            ax_intensity.errorbar(s_data, y_data, yerr=y_err, fmt='o', color="tab:orange", label="Data", markersize=4)
            ax_intensity.fill_between(s_data, y_data - y_err, y_data + y_err, alpha=0.2, color="tab:orange")
            ax_intensity.set_title(f"{label} Intensity (χ² = {chi2:.1f})")

            ax_residuals.axhline(0, color="black", linestyle="--", linewidth=1)
            ax_residuals.fill_between(s_data, -1, 1, alpha=0.15, color="gray", label=r"$\pm1\sigma$")
            ax_residuals.plot(s_data, residuals, marker='o', linestyle='-', color="tab:green", label="Residuals")
            ax_residuals.set_ylabel("Normalized Residual")
            ax_residuals.set_title(f"{label} Residuals")
            ax_residuals.grid(True)
            ax_residuals.legend()
        else:
            ax_residuals.axis("off")  # no residuals to show

    # Inclusive subplot (if present)
    if has_inclusive:
        label = "inclusive"
        j = channel_list.index(label)
        mask = masks_np[:, j]
        s_data = s_np[mask]
        y_data = values_np[mask, j]
        y_err = errors_np[mask, j]
        y_model = intensity_np[mask].sum(axis=1)

        residuals = (y_model - y_data) / y_err
        chi2 = np.sum(residuals**2)
        chi2_total += chi2

        ax_incl_int = axes[-1, 0]
        ax_incl_res = axes[-1, 1]

        ax_incl_int.plot(s_np, intensity_np.sum(axis=1), label="Model (sum)", color="black", linestyle="--")
        ax_incl_int.errorbar(s_data, y_data, yerr=y_err, fmt='o', color="tab:orange", label="Data", markersize=4)
        ax_incl_int.fill_between(s_data, y_data - y_err, y_data + y_err, alpha=0.2, color="tab:orange")
        ax_incl_int.set_title(f"Inclusive Intensity (χ² = {chi2:.1f})")
        ax_incl_int.set_ylabel("Intensity")
        ax_incl_int.grid(True)
        ax_incl_int.legend()

        ax_incl_res.axhline(0, color="black", linestyle="--", linewidth=1)
        ax_incl_res.fill_between(s_data, -1, 1, alpha=0.15, color="gray", label=r"$\pm1\sigma$")
        ax_incl_res.plot(s_data, residuals, marker='o', linestyle='-', color="tab:green", label="Residuals")
        ax_incl_res.set_ylabel("Normalized Residual")
        ax_incl_res.set_title(f"Inclusive Residuals")
        ax_incl_res.grid(True)
        ax_incl_res.legend()

    plt.xlabel(r"$\sqrt{s}$")
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    print(f"Total χ² (summed across all data channels): {chi2_total:.2f}")
    plt.show()


import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Necessario per il plotting 3D

def plot_complex_tensor_columns(x: torch.Tensor, y: torch.Tensor, labels=None):
    """
    Plotta ciascuna colonna di un tensore complesso y (shape: [N, C]) rispetto a un tensore complesso x (shape: [N]).
    Il grafico è un plot 3D con assi:
        - asse x: Re(x)
        - asse y: Im(x)
        - asse z: |y|
    I punti sono colorati in base alla fase di y.
    
    Args:
        x (torch.Tensor): Tensore complesso 1D di shape [N].
        y (torch.Tensor): Tensore complesso 2D di shape [N, C].
        labels (list of str, optional): Etichette per ciascuna colonna. Se None,
            le colonne saranno etichettate come 'Colonna 1', 'Colonna 2', ..., 'Colonna C'.
    """
    # Assicurati che i tensori siano su CPU e staccati dal grafo computazionale
    x = x.detach().cpu()
    y = y.detach().cpu()
    
    # Validazione delle shape dei tensori
    if x.ndim != 1 or y.ndim != 2:
        raise ValueError("x deve essere un tensore 1D e y deve essere un tensore 2D.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x e y devono avere lo stesso numero di righe.")
    if not torch.is_complex(x):# or not torch.is_complex(y):
        raise ValueError("x deve essere tensori complessi.")
    
    N, C = y.shape

    # Preparazione delle etichette
    if labels is not None:
        if len(labels) != C:
            raise ValueError(f"Il numero di etichette ({len(labels)}) non corrisponde al numero di colonne in y ({C}).")
    else:
        labels = [f'Colonna {i+1}' for i in range(C)]
    
    # Estrazione delle parti reali e immaginarie di x
    x_real = x.real.numpy()
    x_imag = x.imag.numpy()
    
    # Creazione dei plot 3D per ciascuna colonna in y
    for i in range(C):
        y_abs = y[:, i].abs().numpy()
        y_phase = torch.angle(y[:, i]).numpy()  # Fase di y in radianti
        
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(x_real, x_imag, y_abs, c=y_phase, cmap='hsv', s=20)
        
        ax.set_xlabel('Re(x)')
        ax.set_ylabel('Im(x)')
        ax.set_zlabel(f'|{labels[i]}|')
        ax.set_title(f'3D Plot: Re(x), Im(x), |{labels[i]}| con colore basato sulla fase')
        
        cbar = fig.colorbar(scatter, ax=ax, label='Fase di y (radiani)')
        cbar.set_ticks([-torch.pi, -torch.pi/2, 0, torch.pi/2, torch.pi])
        cbar.set_ticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
        
        plt.tight_layout()
        plt.show()







