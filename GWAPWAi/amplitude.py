import scipy
import numpy as np
import time
from scipy.integrate import quad
import matplotlib.pyplot as plt
import torch
import torch.special
import math

def threshold(masses: torch.Tensor):
    return masses.sum(dim=1)**2
def momentum(s: torch.Tensor, masses: torch.Tensor) -> torch.Tensor:
    s = torch.atleast_1d(s).to(dtype=torch.cdouble)
    masses = torch.atleast_2d(masses).to(dtype=torch.cdouble)
    threshold_sq = masses.sum(dim=1)**2  # shape (M,)
    return 0.5 * torch.sqrt(s[:, None] - threshold_sq[None, :])

def true_momentum(s:torch.Tensor,masses:torch.Tensor):
    s = torch.atleast_1d(s)           # s becomes shape (N,) even if it was scalar.
    masses = torch.atleast_2d(masses)
    return 0.5*torch.sqrt(s.unsqueeze(1)-masses.sum(dim=1).unsqueeze(0)**2)


#s should be a torch tensor containing all values of s at which we're evaluating. 
#Returns a tensor of the same shape as the input
def omega_pole(s,s0):
    return s/(s+s0)
def omega_scaled(s,smin,smax):
    return -torch.ones(s.shape)+2*(s-smin)/(smax-smin)
def omega_pole_scaled(s, s0, smin, smax):
    return torch.ones(s.shape)+2*(omega_pole(s,s0)-omega_pole(smin,s0))/(omega_pole(smin,s0)-omega_pole(smax,s0))
def chebyshev_t(n, s):
    # s must be in [-1,1] for a real result.
    return torch.cos(n * torch.acos(s))
def numerator(s:torch.Tensor, coeff:torch.Tensor)->torch.Tensor:
    """
    Computes S_m(s) = sum_{n=0}^{N-1} coeff[m, n] * T_n(s) for each set m.

    Parameters:
        coeff (Tensor): A 2D tensor of shape (M, N) where each row m contains coefficients.
        s (Tensor): A tensor of evaluation points (can be any shape).

    Returns:
        Tensor: A tensor of shape (M, *s.shape) where each row m is the series evaluated at s.
    """
    M, N = coeff.shape
    degrees = torch.arange(N, device=coeff.device)
    Tns = torch.stack([chebyshev_t(n, s) for n in degrees], dim=0)
    
    # Reshape coeff to (M, N, 1, ..., 1) so it can broadcast with Tns.
    # s.ndim gives the number of dimensions in s.
    coeff_reshaped = coeff.view(M, N, *([1] * s.ndim))
    
    # Multiply the coefficients with the corresponding T_n(s) and sum over n (the Chebyshev index).
    result = (coeff_reshaped * Tns.unsqueeze(0)).sum(dim=1)
    return result.T

def rhoNnominal(sprime: torch.Tensor, masses: torch.Tensor, momenta: callable, J: int, alpha: float, sL: float, sheet: int) -> torch.Tensor:
    """
    Computes the nominal rhoN function as a 1d tensor
    
    For each scalar s' in sprime, it returns a vector of length num_p whose entries are:
    
        (2 * p[j])^(2*J+1) / (s' + sL)^(2*J+alpha)
    
    Parameters:
      sprime : torch.Tensor of shape (N,), the energy squared values (s').
      masses : torch.Tensor of shape (M,), the masses of the channels.
      momenta: callable, a function that takes a tensor of energy squared values and a tensor of masses and returns a tensor of momenta.
      J      : Scalar, the angular momentum.
      alpha  : Scalar.
      sL     : Scalar.
    
    Returns:
      A torch.Tensor of shape (N, M) where,
      result[i,j] = (2 * p[j])^(2*J+1) / (sprime[i] + sL)^(2*J+alpha)
    """
    sprime = torch.atleast_1d(sprime)# s becomes shape (N,) even if it was scalar.
    p = torch.atleast_1d(momenta(sprime,masses)) #shape (N,M)
    # Compute numerator: (2 * p_i)^(2J+1) for each channel
    num = (2 * p) ** (2 * J + 1)  # shape: (N,M)
    # Compute denominator for each s' value: (s' + sL)^(2J+alpha)
    # Unsqueeze sprime to shape (N, 1) so broadcasting works.
    denom = ((sprime + sL) ** (J + alpha)).unsqueeze(1)  # shape: (N, 1)
    # Calculate diagonal values for each s'
    return num / denom  # shape: (N,M)

def K_nominal(s: torch.Tensor, couplings: torch.Tensor, m_rs: torch.Tensor, bkground: torch.Tensor) -> torch.Tensor:

    """
    Compute the K matrix for each energy s, vectorized over s.

    Parameters:
      s            : Scalar or 1D tensor of complex energies (shape: (N,) if vectorized).
      couplings    : Tensor of shape (numChannels, numRes), complex couplings.
      m_rs         : Tensor of shape (numRes,), complex resonance masses.
      bkground     : Tensor of shape (order, numChannels, numChannels), complex polynomial coefficients.
    
    Returns:
      kmat         : Tensor of shape (N, numChannels, numChannels) containing the K matrix for each s.
    """
    s = torch.atleast_1d(s)
    
    N = s.shape[0]  # Number of s values.
    numChannels, numRes = couplings.shape
    order = bkground.shape[0]
    couplings = couplings.to(dtype=torch.cdouble, device=s.device)
    # Reshape s to (N, 1) so that subtraction broadcasts: (N, 1) - (numRes,) -> (N, numRes).
    s_expanded = s.view(N, 1)
    F = 1.0 / (m_rs - s_expanded)  # Shape: (N, numRes)

    term_matrix = couplings.unsqueeze(0) * F.unsqueeze(1)  # Shape: (N, numChannels, numRes)
    kmat = term_matrix@couplings.T            # Shape: (N, numChannels, numChannels)

    j_exponents = torch.arange(order, device=s.device, dtype=s.real.dtype)  # Shape: (order,)

    poly_factors = s.unsqueeze(1) ** j_exponents  # Shape: (N, order)
    poly_sum = (poly_factors.view(N, order, 1, 1) * bkground.unsqueeze(0)).sum(dim=1)  # Shape: (N, numChannels, numChannels)

    kmat = kmat + poly_sum

    return kmat

def integrate_rhoN(s: torch.Tensor,
                               masses: torch.Tensor,
                               momenta: callable,
                               rhoN: callable,
                               J: int,
                               alpha: float,
                               sL: float,
                               sheet: int,
                               epsilon: float,
                               num_integ: int) -> torch.Tensor:
    """
    Computes the integrated function I(s) to infinity for each energy and channel.
    
    For each energy value s[n] (n=1..N) and for each channel m (m=1..M), it computes:
    
        I(s[n], m) = (s[n]/π) * ∫₀¹  { ρₙₙₒₘᵢₙₐₗ(s', p(s'), J, alpha, sL, sheet) 
                                 / [ s' * (s' - s[n] - iε) ] } 
                      × [1/(1-t)²] dt
                      
    where the substitution is
        s' = L[m] + t/(1-t)
    and L[m] = threshold(masses)[m].
    
    Parameters:
      s         : Tensor of shape (N,), energy values.
      num_integ : Integer, number of integration points (higher → more precision).
      masses    : Tensor of shape (M, num_masses), masses for each channel.
      momenta   : callable, a function that takes a tensor of energy squared values and a tensor of masses and returns a tensor of momenta.
      J, alpha, sL, sheet, epsilon : Parameters for ρₙₙₒₘᵢₙₐₗ and the integrand.
    
    Returns:
      Tensor of shape (N, M) containing the integrated values.
    """
    s = torch.atleast_1d(s) # shape:(N,)
    N = s.shape[0]
    
    lower_limits = threshold(masses) #shape:(M,)
    M = lower_limits.shape[0]

    t_max = 0.9999999
    
    # Create an integration grid parameter t ∈ [0,1] with num_integ points.
    t = torch.linspace(0, t_max, steps=num_integ, device=s.device, dtype=s.dtype)  #shape:(num_integ,)
    sprime_grid = lower_limits.unsqueeze(0) + t.unsqueeze(1)/(1-t).unsqueeze(1) #shape:(num_integ,M)
    jacobian = 1.0/(1-t).unsqueeze(1)**2 #shape:(num_integ,1)
  
    F = rhoN(sprime_grid, masses, momenta, J, alpha, sL, sheet).squeeze(1)  #shape:(num_integ, M)
    
    s_exp   = s.unsqueeze(0).unsqueeze(2)         # shape: (1, N, 1)
    sprime_exp = sprime_grid.unsqueeze(1)          # shape: (num_integ, 1, M)
    factor = (s_exp / math.pi) / (sprime_exp * (sprime_exp - s_exp - 1j * epsilon))#shape:(num_integ, N, M)
    
    integrand = F.unsqueeze(1) * factor * jacobian.unsqueeze(1)          # shape: (num_integ, N, M)
    
    # Expand the x-values for integration: sprime_grid from (num_integ, M) to (num_integ, 1, M)
    x_vals = t.unsqueeze(1).unsqueeze(2)              # shape: (num_integ, 1, M)
    I_integrated = torch.trapezoid(integrand, x=x_vals, dim=0)  # shape: (N, M)
    
    return I_integrated

def integrate_rhoN_scp(s: torch.Tensor,
                   masses: torch.Tensor,
                   momenta: callable,
                   rhoN: callable,
                   J: int,
                   alpha: float,
                   sL: float,
                   sheet: int,
                   epsilon: float,
                   num_integ: int) -> torch.Tensor:
    """
    Computes the integrated function I(s) for each energy and channel using SciPy’s adaptive quadrature.
    
    For each energy value s[n] (n=1..N) and for each channel m (m=1..M):
    
        I(s[n], m) = (s[n]/π) * ∫₀^(t_max) { rhoN(s', masses, momenta, J, alpha, sL, sheet) / 
                        [ s' * (s' - s[n] - iε) ] } * [1/(1-t)²] dt,
                        
    with the substitution
        s' = L[m] + t/(1-t)   and   L[m] = threshold(masses)[m],
    and t_max is chosen close to 1 (here 0.999999) to avoid the singularity at t=1.
    
    Parameters:
      s         : Tensor of shape (N,), energy values.
      num_integ : Maximum number of subintervals (passed to quad’s limit).
      masses    : Tensor of shape (M, num_masses), masses for each channel.
      momenta   : Callable that computes momenta given energy-squared values and masses.
      rhoN      : Callable; function to compute ρ_N given s′, masses, momenta, J, alpha, sL, sheet.
      J, alpha, sL, sheet, epsilon : Parameters passed to rhoN and the integrand.
      
    Returns:
      A torch.Tensor of shape (N, M) containing the integrated values (complex).
    """
    # Ensure s is at least 1D.
    s = torch.atleast_1d(s)  # shape: (N,)
    N_energy = s.shape[0]
    
    # Obtain the channel lower limits (assumed defined by threshold).
    lower_limits = threshold(masses)  # shape: (M,)
    M_channels = lower_limits.shape[0]
    
    # Define the upper integration limit in t to avoid t=1.
    t_max = 0.9999999999
    
    # Prepare a list for storing results.
    results = []
    
    # Loop over each energy value.
    for s_val in s:
        s_val_float = s_val.item()# may be complex
        channel_vals = []
        # Loop over channels.
        for m_idx in range(M_channels):
            L_val = lower_limits[m_idx].item()
            # Define the integrand as a function of t (a scalar float).
            
            s_val_tensor = torch.tensor([np.real(s_val_float)], dtype=torch.float64, device=s.device)
            F_full_ext = rhoN(s_val_tensor, masses, momenta, J, alpha, sL, sheet)
            F_val_ext = F_full_ext[0, m_idx].item()  # may be complex
            
            def altern_integrand(t):
                sprime = L_val + t/(1-t)
                sprime_tensor = torch.tensor([sprime], dtype=torch.float64)
                F_full = rhoN(sprime_tensor, masses, momenta, J, alpha, sL, sheet)
                F_val = F_full[0, m_idx].item()  # may be complex

                jac = 1.0/(1-t)**2
                return (F_val - np.real(F_val_ext)) / (sprime * (sprime - np.real(s_val_float))) * jac
            
            def integrand(t):
                # Transformation: s' = L + t/(1-t)
                sprime = L_val + t/(1-t)
                # Convert sprime to a one-element torch tensor (float64) for rhoN.
                sprime_tensor = torch.tensor([sprime], dtype=torch.float64, device=s.device)
                # Call the passed-in rhoN function.
                # It should return a tensor of shape (1, M). We extract the value for channel m_idx.
                F_full = rhoN(sprime_tensor, masses, momenta, J, alpha, sL, sheet)
                F_val = F_full[0, m_idx].item()  # may be complex
                # The Jacobian from t->s': 1/(1-t)^2.
                jac = 1.0/(1-t)**2
                return F_val / (sprime * (sprime - s_val_float - 1j*epsilon)) * jac
            
            # Integrate the real part.#
            if abs(np.imag(s_val)) < 2. * epsilon:
                real_int, real_err = quad(lambda t: np.real(altern_integrand(t)),
                                          0, t_max, limit=num_integ, epsabs=1e-12)
                imag_int, imag_err = quad(lambda t: np.imag(altern_integrand(t)),
                                          0, t_max, limit=num_integ, epsabs=1e-12)
                I_val = (np.real(s_val_float)/math.pi) * (real_int + 1j*imag_int) + np.real(F_val_ext) * np.log(L_val/(abs(np.real(s_val_float) - L_val)))/math.pi
                #print(L_val, np.real(s_val_float), np.log(L_val/(abs(np.real(s_val_float) - L_val))))
                if np.real(s_val_float) > L_val:
                    I_val += 1j*np.real(F_val_ext)
            else:
                real_int, real_err = quad(lambda t: np.real(integrand(t)),
                                          0, t_max, limit=num_integ, epsabs=1e-12)
                # Integrate the imaginary part.
                imag_int, imag_err = quad(lambda t: np.imag(integrand(t)),
                                          0, t_max, limit=num_integ, epsabs=1e-12)
                # Combine and multiply by the overall factor s/π.
                I_val = (s_val_float/math.pi) * (real_int + 1j*imag_int)
            channel_vals.append(I_val)
        results.append(channel_vals)
    
    # Convert the list of lists to a numpy array and then to a torch tensor.
    results_np = np.array(results, dtype=np.complex128)  # shape: (N, M)
    return torch.tensor(results_np)


def get_true_momentum(s: torch.Tensor, masses: torch.Tensor) -> torch.Tensor:
    """
    Computes the "true momentum" according to:
    
      p(s) = 0.5 * sqrt((s - (m1+m2)**2) * (s - (m1-m2)**2)) / s
      
    This function is vectorized over s. It expects s to be a tensor of shape (N,)
    and masses to be a tensor of shape (2,).
    """
    # Ensure s is at least 1D and masses is a tensor.
    s = torch.atleast_1d(s)
    masses = torch.as_tensor(masses, dtype=torch.cdouble, device=s.device)
    
    m_sum = masses[0] + masses[1]
    m_diff = masses[0] - masses[1]
    
    # Compute (m1+m2)^2 and (m1-m2)^2
    m_sum_sq = m_sum**2
    m_diff_sq = m_diff**2
    
    # Compute numerator: sqrt((s - (m1+m2)^2)*(s - (m1-m2)^2))
    num = torch.sqrt((s - m_sum_sq) * (s - m_diff_sq))
    
    return 0.5 * num / s


def safe_power(x: torch.Tensor, power: float, eps: float = 1e-12) -> torch.Tensor:
    """
    Computes x**power with gradient stability near |x| ~ 0.
    Prevents nan or inf in PowBackward0 during autograd.

    Parameters:
        x     : Tensor (real or complex)
        power : Exponent (float)
        eps   : Stability cutoff

    Returns:
        Tensor of x ** power with safe handling near zero
    """
    x_mag = torch.abs(x)
    x_safe = torch.where(x_mag < eps, x.new_full(x.shape, eps, dtype=x.dtype), x)
    return x_safe ** power

def construct_phsp(s: torch.Tensor, masses: torch.Tensor, J: float) -> torch.Tensor:
    """
    Computes diagonalized phase-space factors safely for all channels.

    Parameters:
        s       : Tensor of shape (N,), complex dtype
        masses  : Tensor of shape (M, 2), real or complex
        J       : Angular momentum

    Returns:
        Tensor of shape (N, M, M) — diagonal matrix per point
    """
    s = s.view(-1).to(dtype=torch.cdouble)
    masses = masses.to(dtype=torch.cdouble)

    m1 = masses[:, 0]
    m2 = masses[:, 1]

    m_sum_sq = (m1 + m2) ** 2
    m_diff_sq = (m1 - m2) ** 2

    s_exp = s[:, None]  # (N, 1)

    k_squared = (s_exp - m_sum_sq) * (s_exp - m_diff_sq)
    k = torch.sqrt(k_squared + 0j) / (2 * s_exp + 0j)  # safe sqrt

    phsp = safe_power(k, J + 0.5) / safe_power(s_exp, 0.25)
    return torch.diag_embed(phsp)



def precompute_II(s, static_params, integrate_rhoN_func, rhoN_dispatcher, momentum_func, sheet = 0, epsilon = 0.003, num_integ = 100000):
    J = static_params["J"]
    alpha = static_params["alpha"]
    sL = static_params["sL"]
    masses = static_params["masses"]
    rhoN_fn = rhoN_dispatcher[static_params["rhoNtype"]]

    # Compute and return diagonalized II matrix
    return torch.diag_embed(integrate_rhoN_func(s, masses, momentum_func, rhoN_fn, J, alpha, sL, sheet, epsilon, num_integ))


def compute_intensity(s, flat_params, unpack_fn, static_params,
                      II,
                      numerator_func, omega_pole_func, 
                      construct_phsp_func, K_nominal_func, 
                      momentum_func):
    chebys, pole_couplings, pole_masses, kbkgrd = unpack_fn(flat_params)

    J = static_params["J"]
    s0 = static_params["s0"]
    masses = static_params["masses"]
    num_ch = static_params["num_ch"]

    N = numerator_func(omega_pole_func(s, s0), chebys).unsqueeze(1).to(torch.cdouble)
    phsp = construct_phsp_func(s, masses, J).to(torch.cdouble)
    kmat = K_nominal_func(s, pole_couplings, pole_masses, kbkgrd)

    num_pts = s.shape[0]
    identity = torch.eye(num_ch, dtype=s.dtype, device=s.device).expand(num_pts, num_ch, num_ch)

    #regularize the denom
    reg = 1e-6 * torch.eye(kmat.shape[-1], dtype=kmat.dtype, device=kmat.device).expand_as(kmat)
    denom_inv = (identity - kmat @ II + reg).inverse() @ kmat

    #denom_inv = (identity - kmat @ II).inverse() @ kmat

    val = (N @ denom_inv @ phsp).squeeze(1)
    intensity = (val * torch.conj(val)).real

    return intensity


# Dispatcher map defined once and shared
rhoN_dispatcher = {
    "rhoN-nominal": rhoNnominal
}


