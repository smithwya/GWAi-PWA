#!/usr/bin/env python3
import re
import ast
import pprint
import numpy as np
import torch
def parse_line(line):
    """
    Extracts the command and its argument string from a line of the form:
         Command(arg1, arg2, ...)
    """
    line = line.strip()
    if not line:
        return None, None
    # Use a regex that finds a word followed by parentheses
    m = re.match(r'(\w+)\((.*)\)', line)
    if not m:
        return None, None
    cmd = m.group(1)
    args_str = m.group(2)
    return cmd, args_str

def split_args(s):
    """
    Splits an argument string into individual argument tokens.
    It takes care not to split inside braces or quotes.
    """
    args = []
    current = ""
    depth = 0
    in_quote = False
    quote_char = ""
    for char in s:
        if char in "\"'":
            if in_quote:
                if char == quote_char:
                    in_quote = False
            else:
                in_quote = True
                quote_char = char
        if not in_quote:
            if char in "{([":
                depth += 1
            elif char in ")}]":
                depth -= 1
        if char == ',' and depth == 0 and not in_quote:
            args.append(current.strip())
            current = ""
        else:
            current += char
    if current:
        args.append(current.strip())
    return args

def parse_token(token):
    """
    Recursively converts a token string into a Python object.
    - If the token is enclosed in {} it is treated as a list.
    - Numeric tokens are converted to int/float.
    - Tokens containing "\pm" or "±" are split into a tuple (value, uncertainty).
    - "Yes"/"No" are converted to booleans.
    - Quoted strings are unquoted.
    """
    token = token.strip()
    # If token is a braced list, parse its contents recursively.
    if token.startswith("{") and token.endswith("}"):
        inner = token[1:-1].strip()
        items = split_args(inner)
        parsed_items = [parse_token(item) for item in items]
        # If all items are numbers (or complex numbers) then return a tensor.
        if all(isinstance(item, (int, float, complex)) for item in parsed_items):
            return torch.tensor(parsed_items, dtype=torch.float64)
        # If items are tuples (value, error), convert separately.
        if all(isinstance(item, tuple) and len(item)==2 and all(isinstance(x, (int, float, complex)) for x in item) 
               for item in parsed_items):
            centers = torch.tensor([item[0] for item in parsed_items], dtype=torch.float64)
            errors  = torch.tensor([item[1] for item in parsed_items], dtype=torch.float64)
            return (centers, errors)
        # Otherwise, return the list.
        return parsed_items

    # Check for the ± (or \pm) symbol.
    if r'\pm' in token or '±' in token:
        token = token.replace(r'\pm', '±')
        parts = token.split('±')
        try:
            value = float(parts[0].strip())
            error = float(parts[1].strip())
            return (value, error)
        except Exception:
            return token

    # Try to convert to int then float.
    try:
        return int(token)
    except Exception:
        try:
            return float(token)
        except Exception:
            pass

    # Convert boolean words.
    if token == "Yes":
        return True
    if token == "No":
        return False

    # Remove quotes if present.
    if (token.startswith('"') and token.endswith('"')) or (token.startswith("'") and token.endswith("'")):
        return token[1:-1]

    return token

def parse_arguments(args_str):
    """
    Splits the arguments string and parses each token.
    """
    tokens = split_args(args_str)
    return [parse_token(token) for token in tokens]

def parse_file(filename):
    """
    Reads the file and stores each instruction in a dictionary.
    For commands that occur multiple times (like AddChannel), the value is
    stored as a list of argument lists.
    """
    instructions = {}
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            cmd, args_str = parse_line(line)
            if cmd is None:
                continue
            parsed_args = parse_arguments(args_str)
            if cmd in instructions:
                # If already a list, append the new occurrence.
                if isinstance(instructions[cmd], list) and isinstance(instructions[cmd][0], list):
                    instructions[cmd].append(parsed_args)
                else:
                    instructions[cmd] = [instructions[cmd], parsed_args]
            else:
                instructions[cmd] = parsed_args
    return instructions

def reorganize_instructions(instr):
    import torch

    def to_number(x):
        if isinstance(x, torch.Tensor) and x.ndim == 0:
            return x.item()
        return x

    data = {}

    # --- Channels ---
    chans_raw = instr.get("AddChannel", [])
    if not isinstance(chans_raw[0], list):
        chans_raw = [chans_raw]
    channel_names = [ch[0] for ch in chans_raw]
    channel_masses = [ch[1] for ch in chans_raw]
    masses_tensor = torch.stack(channel_masses, dim=0)
    data["channels"] = {"names": channel_names, "masses": masses_tensor}

    # --- Waves ---
    waves_raw = instr.get("AddWave", [])
    if not isinstance(waves_raw[0], list):
        waves_raw = [waves_raw]
    waves = []
    wave_params = {}
    for entry in waves_raw:
        wave = entry[0]
        waves.append(wave)
        wave_params[wave] = {
            "kmat_type": entry[1],
            "rhoN_type": entry[2],
            "J": entry[3],
            "sL": entry[4],
        }
    data["waves"] = waves

    # --- Wave Data ---
    data["wave_data"] = {}
    for wave in waves:
        wd = wave_params[wave].copy()

        # --- Chebyshev Coefficients ---
        cheby = instr.get("ChebyCoeffs", [])
        if not isinstance(cheby[0], list):
            cheby = [cheby]
        s_list, coeffs_list, errs_list = [], [], []
        for ch in channel_names:
            found = False
            for entry in cheby:
                if entry[0] == wave and entry[1] == ch:
                    found = True
                    s_list.append(entry[2])
                    raw = entry[3]
                    if isinstance(raw, tuple):
                        coeffs_list.append(raw[0])
                        errs_list.append(raw[1])
                    else:
                        coeffs = []
                        errs = []
                        for item in raw:
                            if isinstance(item, tuple):
                                coeffs.append(to_number(item[0]))
                                errs.append(to_number(item[1]))
                            else:
                                coeffs.append(to_number(item))
                                errs.append(0.0)
                        coeffs_list.append(torch.tensor(coeffs, dtype=torch.float64))
                        errs_list.append(torch.tensor(errs, dtype=torch.float64))
                    break
            if not found:
                s_list.append("")
                coeffs_list.append(torch.tensor([], dtype=torch.float64))
                errs_list.append(torch.tensor([], dtype=torch.float64))
        wd["ChebyCoeffs"] = {
            "s": s_list,
            "coeffs": torch.stack(coeffs_list),
            "errors": torch.stack(errs_list),
        }

        # --- Poles ---
        poles = instr.get("AddPole", [])
        if not isinstance(poles[0], list):
            poles = [poles]
        pmasses, pmerrs, pcouplings, pcoup_errs = [], [], [], []
        for entry in poles:
            if entry[0] != wave:
                continue
            pmass, pmerr = entry[1]
            chnames = entry[2]
            raw = entry[3]
            if isinstance(raw, tuple):
                coup, errs = list(raw[0]), list(raw[1])
            else:
                coup, errs = [], []
                for item in raw:
                    if isinstance(item, tuple):
                        coup.append(to_number(item[0]))
                        errs.append(to_number(item[1]))
                    else:
                        coup.append(to_number(item))
                        errs.append(0.0)
            full = [0.0] * len(channel_names)
            full_err = [0.0] * len(channel_names)
            for c, v, e in zip(chnames, coup, errs):
                idx = channel_names.index(c)
                full[idx] = v
                full_err[idx] = e
            pmasses.append(pmass)
            pmerrs.append(pmerr)
            pcouplings.append(full)
            pcoup_errs.append(full_err)
        if pmasses:
            # Transpose couplings to get shape (num_channels, num_resonances)
            pcouplings_tensor = torch.tensor(pcouplings).T
            pcoup_errs_tensor = torch.tensor(pcoup_errs).T
            wd["Poles"] = {
                "mass": torch.tensor(pmasses),
                "mass_err": torch.tensor(pmerrs),
                "couplings": pcouplings_tensor,
                "coupling_err": pcoup_errs_tensor,
            }
        else:
            wd["Poles"] = None

        # --- K-Matrix Background ---
        kbacks = instr.get("AddKmatBackground", [])
        if not isinstance(kbacks[0], list):
            kbacks = [kbacks]
        bgdict = {}
        for entry in kbacks:
            if entry[0] != wave:
                continue
            order = entry[1]
            raw_rows = entry[2]
            rows = []
            for row in raw_rows:
                if isinstance(row, tuple):
                    vals = row[0].tolist()
                elif isinstance(row, list):
                    vals = []
                    for item in row:
                        vals.extend(item[0].tolist() if isinstance(item, tuple) else [to_number(item)])
                else:
                    vals = [to_number(row)]
                padded = vals + [0.0] * (len(channel_names) - len(vals))
                rows.append(padded[:len(channel_names)])
            bgdict[order] = torch.tensor(rows)
        if bgdict:
            orders = sorted(bgdict)
            wd["KmatBackground"] = torch.stack([bgdict[o] for o in orders])
        else:
            wd["KmatBackground"] = None

        # --- LoadExpData ---
        exp_entries = instr.get("LoadExpData", [])
        if not isinstance(exp_entries[0], list):
            exp_entries = [exp_entries]
        exp_dict = {}
        for entry in exp_entries:
            if entry[0] == wave:
                exp_dict[entry[1]] = entry[2]
        wd["LoadExpData"] = exp_dict

        data["wave_data"][wave] = wd

    # --- Merge inclusive data into wave_data[w]["LoadExpData"]["inclusive"] ---
    incl_entries = instr.get("LoadExpInclusiveCrossSection", [])
    if not isinstance(incl_entries, list) or (incl_entries and not isinstance(incl_entries[0], list)):
        incl_entries = [incl_entries]
    for entry in incl_entries:
        if len(entry) == 2:
            wave, fname = entry
            if wave not in data["wave_data"]:
                data["wave_data"][wave] = {}
            if "LoadExpData" not in data["wave_data"][wave]:
                data["wave_data"][wave]["LoadExpData"] = {}
            data["wave_data"][wave]["LoadExpData"]["inclusive"] = fname

    # --- Global Fitting Parameters ---
    global_keys = [
        "SetSeed", "FitRegion", "FittingSequence", "FittingParameters",
        "InclChi2Weight", "ExclChi2Weight", "ReducedChi2CutOff",
        "ChooseAnAction", "DoRandomize", "IncludeAlsoInclusiveCrossSection",
        "PolesearchGrid", "PolesearchZero"
    ]
    data["fitting"] = {k: instr[k] for k in global_keys if k in instr}

    return data


def load_experimental_data(config):
    exp_data = {}

    for wave, wdata in config.get("wave_data", {}).items():
        wave_block = {
            "channels": {},
            "inclusive": None
        }
        files = wdata.get("LoadExpData", {})

        for name, fname in files.items():
            try:
                data_np = np.loadtxt(fname)
                tensor = torch.tensor(data_np, dtype=torch.float64)
                if name == "inclusive":
                    wave_block["inclusive"] = tensor
                else:
                    wave_block["channels"][name] = tensor
            except Exception as e:
                print(f"Warning: Failed to load '{fname}' for wave '{wave}', name '{name}': {e}")
                if name == "inclusive":
                    wave_block["inclusive"] = None
                else:
                    wave_block["channels"][name] = None

        exp_data[wave] = wave_block

    return exp_data

