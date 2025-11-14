# Copyright 2025 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module contains building blocks for cropping."""

import logging
from typing import NamedTuple

import numpy as np
from biotite.structure import Atom, AtomArray, get_chain_starts
from scipy.spatial.distance import cdist

from openfold3.core.data.primitives.quality_control.logging_utils import (
    log_runtime_memory,
)
from openfold3.core.data.primitives.structure.interface import (
    get_interface_atoms,
    get_query_interface_atom_pair_idxs,
    get_query_interface_token_center_atoms,
)
from openfold3.core.data.primitives.structure.labels import (
    assign_atom_indices,
    remove_atom_indices,
)
from openfold3.core.data.resources.residues import MoleculeType

NO_CROPPING_TOKEN_BUDGET_SENTINEL = -1

logger = logging.getLogger(__name__)


def get_small_ligand_chain_ids(
    atom_array: AtomArray,
    max_atoms: int = 5,
) -> np.ndarray:
    """Returns chain IDs of small ligands

    The "smallness" of a ligand is determined by the number of atoms it consists of.

    Args:
        atom_array:
            AtomArray containing the structure to get the small ligand chain IDs from.
        max_atoms:
            Maximum number of atoms for a ligand to be considered small. Default is 5.

    Returns:
        Array of unique chain IDs of small ligands.
    """
    ligand_array = atom_array[atom_array.molecule_type_id == MoleculeType.LIGAND]

    if len(ligand_array) == 0:
        return np.array([])

    lig_chain_starts = get_chain_starts(ligand_array, add_exclusive_stop=True)
    lig_chain_sizes = np.diff(lig_chain_starts)
    small_lig_chain_idxs = np.where(lig_chain_sizes <= max_atoms)[0]
    small_lig_chain_ids = ligand_array[lig_chain_starts[small_lig_chain_idxs]].chain_id

    # TODO: Dev-only, remove later
    assert np.unique(small_lig_chain_ids).size == small_lig_chain_ids.size, (
        "Small ligand chain IDs are not unique"
    )

    return small_lig_chain_ids


def get_small_ligand_mask(
    atom_array: AtomArray,
    max_atoms: int = 5,
) -> np.ndarray:
    """Returns a mask for atoms in small ligands.

    The "smallness" of a ligand is determined by the number of atoms it consists of.

    Args:
        atom_array:
            AtomArray containing the structure to get the small ligand mask for.
        max_atoms:
            Maximum number of atoms for a ligand to be considered small. Default is 5.
    """
    small_lig_chain_ids = get_small_ligand_chain_ids(atom_array, max_atoms)
    return np.isin(atom_array.chain_id, small_lig_chain_ids)


def precrop_chains(
    atom_array: AtomArray,
    n_chains: int = 20,
    preferred_chain_or_interface: str | list[str, str] | None = None,
    interface_distance_threshold: float = 15.0,
    ignore_ligands_below: int | float = 6,
    ligand_inclusion_distance: float = 5.0,
) -> AtomArray:
    """Subsets structures with too many chains to N chains

    Follows 2.5.4 of the AlphaFold3 SI. Will select a random interface token center atom
    and return the closest N chains based on minimum distances between any token center
    atoms, therefore acting as an initial fixed precropping of too-large assemblies.

    Requires the 'token_center_atom' annotation created by the tokenizer function.

    Args:
        atom_array:
            AtomArray containing the structure to subset
        n_chains:
            Number of chains to keep in the precrop. If the structure has less than N
            chains, all of them are kept. Default is 20.
        interface_distance_threshold:
            Distance threshold in Å that an interface token center atom must have to any
            token center atom in another chain to be considered an interface token
            center atom
        ignore_ligands_below:
            Any ligands with fewer atoms than this number will be excluded from
            the precrop chain count. Instead, they will be included based on
            proximity to the selected chains in the subset. Set to float('inf')
            to never count any ligands toward the chain budget. Default is 6.
        ligand_inclusion_distance:
            Distance threshold in Å when adding ligands excluded by
            `ignore_ligands_below` back into the precrop. Any such ligand with
            a distance ≤ this threshold to any retained chain will be included.

    Returns:
        AtomArray with the closest n_chains based on token center atom distances,
        plus any nearby small ligands.
    """
    # Keep pointer to unfiltered AtomArray
    atom_array_orig = atom_array

    # 1) Hold out "small" ligands before selecting chains
    small_ligand_mask = get_small_ligand_mask(
        atom_array_orig, max_atoms=ignore_ligands_below - 1
    )
    atom_array = atom_array_orig[~small_ligand_mask]

    # 2) Get resolved and preferred token center atoms
    token_center_atoms, preferred_token_center_atoms = (
        get_resolved_and_preferred_token_center_atoms(
            atom_array, preferred_chain_or_interface
        )
    )

    # 3) If interface is given, take random reference atom from that specific interface
    #    region, otherwise take random atom from chain
    if isinstance(preferred_chain_or_interface, list):
        preferred_token_center_atoms = get_interface_atoms(
            atom_array=preferred_token_center_atoms,
            distance_threshold=interface_distance_threshold,
        )

    reference_atom = np.random.choice(preferred_token_center_atoms)

    # 4) Compute distances to all token centers
    dists = cdist(
        reference_atom.coord.reshape(1, 3),
        token_center_atoms.coord,
    ).squeeze(0)

    # 5) Sort chain IDs by distance

    # Get indices that sort distances low-to-high
    sort_idx = np.argsort(dists)

    # Chain IDs of all token centers, sorted by distance to the selected atom
    chain_ids_sorted = token_center_atoms.chain_id[sort_idx]

    # Get indices of the *first* occurrence (i.e., closest distance) for each unique
    # chain ID.
    # Sort these indices to get the distance rank of unique chains.
    unique_chain_idxs_sorted = np.sort(
        np.unique(chain_ids_sorted, return_index=True)[1]
    )

    # Select indices corresponding to the N closest unique chains
    closest_idxs = unique_chain_idxs_sorted[:n_chains]

    # Get the actual IDs of the N closest unique chains
    closest_chain_ids = chain_ids_sorted[closest_idxs]

    # 6) Create mask for closest N chains on the original AtomArray
    n_chain_mask = np.isin(atom_array_orig.chain_id, closest_chain_ids)

    # 7) Build initial subset
    atom_array_subset = atom_array_orig[n_chain_mask]

    # 8) Re-include small ligands by proximity
    atom_array_small_lig = atom_array_orig[small_ligand_mask]
    if len(atom_array_small_lig) == 0:
        return atom_array_subset

    _, proximal_small_lig_chains = get_query_interface_atom_pair_idxs(
        query_atom_array=atom_array_small_lig,
        target_atom_array=atom_array_subset,
        distance_threshold=ligand_inclusion_distance,
        return_chain_pairs=True,
    )

    # 9) If none requalify, just return the core subset
    if proximal_small_lig_chains is None:
        return atom_array_subset

    # 10) Otherwise, add those ligand chains back in
    proximal_ids = np.unique(proximal_small_lig_chains[:, 0])
    proximal_mask = np.isin(atom_array_orig.chain_id, proximal_ids)
    logger.debug(
        f"Adding {len(proximal_ids)} small (< {ignore_ligands_below}) ligand chains to "
        "the precrop by proximity."
    )
    final_mask = n_chain_mask | proximal_mask
    return atom_array_orig[final_mask]


def set_contiguous_crop_mask(atom_array: AtomArray, token_budget: int) -> None:
    """Implements Contiguous Cropping from AF3 SI, 2.7.1.

    Uses Algorithm 1 from AF-Multimer section 7.2.1. to update the input biotite
    atom array with added 'crop_mask' annotation in-place. Note: Algorithm 1
    does not work correctly as stated in the AF-Multimer SI, so here we are using
    a fixed version.

    Args:
        atom_array (atom_array):
            Biotite atom array of the first bioassembly of a PDB entry.
        token_budget (int):
            Token budget i.e. total crop size.

    Returns:
        None
    """
    # Assign atom index
    assign_atom_indices(atom_array)

    # Get chain ids and permute
    chains = np.unique(atom_array.chain_id)
    chains = np.random.permutation(chains)

    # Create cropping mask annotation
    atom_array.set_annotation("crop_mask", np.repeat(False, len(atom_array)))

    # Cropping loop
    # "number of tokens selected so far"
    n_added = 0
    # combined length of yet to be cropped chains excluding current"
    n_remaining = len(set(atom_array.token_id))

    for chain_id in chains:
        # Get chain atom array
        atom_array_chain = atom_array[atom_array.chain_id == chain_id]

        # Get chain length
        chain_length = atom_array_chain.token_id[-1] - atom_array_chain.token_id[0] + 1
        n_remaining -= chain_length

        # Sample length of crop for current chain
        crop_size_max = min(token_budget - n_added, chain_length)
        crop_size_min = min(chain_length, max(0, token_budget - n_added - n_remaining))
        crop_size = np.random.randint(crop_size_min, crop_size_max + 1, 1).item()

        n_added += crop_size

        # Sample start of crop for current chain
        crop_start = np.random.randint(0, chain_length - crop_size + 1, 1).item()

        # Get token indices in the crop
        chain_token_ids = np.unique(atom_array_chain.token_id)
        # Slice using the sampled crop start and length for this chain
        crop_token_index_chain = chain_token_ids[crop_start : crop_start + crop_size]
        # Map to atom indices in the full assembly
        crop_atom_index_chain = atom_array_chain[
            np.isin(atom_array_chain.token_id, crop_token_index_chain)
        ]._atom_idx

        # Edit corresponding segment in crop mask
        atom_array.crop_mask[crop_atom_index_chain] = True

    # Remove atom index
    remove_atom_indices(atom_array)


def set_spatial_crop_mask(
    atom_array: AtomArray,
    token_budget: int,
    preferred_chain_or_interface: str | list[str, str] | None = None,
) -> None:
    """Implements Spatial Cropping from AF3 SI, 2.7.2.

    Uses Algorithm 2 from AF-Multimer section 7.2.2. to update the input biotite
    atom array with added 'crop_mask' annotation in-place. Note: we drop the
    index-based distance-untying step from Algorithm 2 (line 1, i * 10^-3 factor)
    because it distorts the distances and results in less convex spatial crops.

    Args:
        atom_array (AtomArray):
            Biotite atom array of the first bioassembly of a PDB entry.
        token_budget (int):
            Total crop size.
        preferred_chain_or_interface (str | list[str, str] | None):
            Optional chain ID or chain ID pair indicating the preferred chain or
            interface, from which reference atoms are selected. Generated by eq. 1 in
            AF3 SI for the weighted PDB dataset.

    Returns:
        None
    """
    # Subset token center atoms to those in the preferred chain/interface if provided
    token_center_atoms, preferred_token_center_atoms = (
        get_resolved_and_preferred_token_center_atoms(
            atom_array, preferred_chain_or_interface
        )
    )

    # Get reference atom
    reference_atom = np.random.choice(preferred_token_center_atoms)

    # Find spatial crop
    set_spatial_crop_mask_with_ref(
        reference_atom, token_center_atoms, token_budget, atom_array
    )


def set_spatial_interface_crop_mask(
    atom_array: AtomArray,
    token_budget: int,
    preferred_chain_or_interface: str | list[str, str] | None = None,
) -> None:
    """Implements Spatial Interface Cropping from AF3 SI, 2.7.3.

    Uses Algorithm 2 from AF-Multimer section 7.2.2. to update the input biotite
    atom array with added 'crop_mask' annotation in-place. Note: we drop the
    index-based distance-untying step from Algorithm 2 (line 1, i * 10^-3 factor)
    because it distorts the distances and results in less convex spatial crops.

    Args:
        atom_array (AtomArray):
            Biotite atom array of the first bioassembly of a PDB entry.
        token_budget (int):
            Total crop size.
        preferred_chain_or_interface (str | list[str, str] | None):
            Optional chain ID or chain ID pair indicating the preferred chain or
            interface, from which reference atoms are selected. Generated by eq. 1 in
            AF3 SI for the weighted PDB dataset.

    Returns:
        None
    """
    # Subset token center atoms to those in the preferred chain/interface if provided
    token_center_atoms, preferred_token_center_atoms = (
        get_resolved_and_preferred_token_center_atoms(
            atom_array, preferred_chain_or_interface
        )
    )

    if len(set(atom_array.chain_id)) > 1:
        # Find interface token center atoms
        preferred_interface_token_center_atoms = get_query_interface_token_center_atoms(
            preferred_token_center_atoms, token_center_atoms
        )

        # Get reference atom
        reference_atom = np.random.choice(preferred_interface_token_center_atoms)
    # Skip interface subsetting if there is only one chain, making the interface spatial
    # crop equivalent to non-interface spatial crop
    else:
        # Get reference atom
        reference_atom = np.random.choice(preferred_token_center_atoms)

    # Find spatial crop
    set_spatial_crop_mask_with_ref(
        reference_atom, token_center_atoms, token_budget, atom_array
    )


def set_whole_crop_mask(atom_array: AtomArray) -> None:
    """Sets the atom array's crop_mask attribute to all True.

    Args:
        atom_array (AtomArray):
            AtomArray of the input assembly to crop.

    Returns:
        None
    """
    atom_array.set_annotation("crop_mask", np.repeat(True, len(atom_array)))


def get_resolved_token_center_atoms(
    atom_array: AtomArray,
) -> AtomArray:
    """Returns the resolved token center atoms in an atom array.

    Args:
        atom_array (AtomArray):
            AtomArray of the input assembly.

    Returns:
        AtomArray:
            AtomArray of resolved token center atoms.
    """
    token_center_atoms = atom_array[atom_array.token_center_atom]

    # Subset to resolved token center atoms
    token_center_atoms = token_center_atoms[token_center_atoms.occupancy > 0]

    if len(token_center_atoms) == 0:
        raise RuntimeError("No resolved token center atoms found.")

    return token_center_atoms


def get_preferred_subset(
    atom_array: AtomArray,
    preferred_chain_or_interface: str | list[str, str],
) -> AtomArray:
    """Returns the subset of atoms in preferred chain or interface.

    Args:
        atom_array (AtomArray):
            Input AtomArray (in this module token center atoms).
        preferred_chain_or_interface (str | list[str, str]):
            Chain ID or chain ID pair indicating the preferred chain or interface from
            which preferred atoms are selected. Generated by eq. 1 in AF3 SI for the
            weighted PDB dataset.

    Returns:
        AtomArray:
            AtomArray of atoms in the preferred region.
    """
    # If chain provided
    if isinstance(preferred_chain_or_interface, str):
        preferred_atoms = atom_array[
            atom_array.chain_id == preferred_chain_or_interface
        ]
    # If interface provided
    elif isinstance(preferred_chain_or_interface, list):
        preferred_atoms = atom_array[
            np.isin(atom_array.chain_id, preferred_chain_or_interface)
        ]
    else:
        raise ValueError(
            f"Invalid preferred_chain_or_interface: {preferred_chain_or_interface}, "
            "has to be str or 2-list."
        )

    return preferred_atoms


def get_resolved_and_preferred_token_center_atoms(
    atom_array: AtomArray,
    preferred_chain_or_interface: str | list[str, str] | None = None,
) -> tuple[AtomArray, AtomArray]:
    """Returns all token center atoms and those in preferred chain or interface.

    Args:
        atom_array (AtomArray):
            Input AtomArray (in this module token center atoms).
        preferred_chain_or_interface (str | list[str, str] | None):
            Optional chain ID or chain ID pair indicating the preferred chain or
            interface, from which reference atoms are selected. Generated by eq. 1 in
            AF3 SI for the weighted PDB dataset.

    Returns:
        tuple[AtomArray, AtomArray]:
            Tuple of (all resolved token center atoms, preferred token center atoms). If
            no preferred chain/interface is provided, or if no resolved atoms are found
            in the preferred chain/interface, the preferred token center atoms are the
            same as all resolved token center atoms.
    """
    token_center_atoms = get_resolved_token_center_atoms(atom_array)

    if preferred_chain_or_interface is not None:
        preferred_token_center_atoms = get_preferred_subset(
            token_center_atoms, preferred_chain_or_interface
        )
    else:
        preferred_token_center_atoms = token_center_atoms

    # If the preferred chain/interface has no resolved atoms, use all resolved token
    # center atoms
    # Note: this will also be the case if a chain or interface is provided that is not
    # in the structure
    if len(preferred_token_center_atoms) == 0:
        logger.warning(
            "No resolved token center atoms found in the preferred chain/interface, "
            "falling back to all resolved token center atoms."
        )
        preferred_token_center_atoms = token_center_atoms

    return token_center_atoms, preferred_token_center_atoms


def set_spatial_crop_mask_with_ref(
    reference_atom: Atom,
    token_center_atoms: AtomArray,
    token_budget: int,
    atom_array: AtomArray,
) -> None:
    """Finds the token_budget nearest neighbors to reference atom and sets crop_mask.

    This function is the underlying primitive to both default and interface spatial
    cropping, which only differ in how the reference atom is selected.

    Args:
        reference_atom (Atom):
            The sampled reference atom around which the spatial crop is created.
        token_center_atoms (AtomArray):
            The set of token center atoms to crop from.
        token_budget (int):
            Crop size.
        atom_array (AtomArray):
            Input atom array of the bioassembly.

    Returns:
        None
    """
    # Get distance from all other token center atoms and break ties
    distances_to_reference_atom = cdist(
        np.reshape(reference_atom.coord, (1, -1)), token_center_atoms.coord
    ).squeeze(0)

    # Get token_budget nearest token center atoms
    nearest_token_center_atom_ids = np.argsort(distances_to_reference_atom)[
        :token_budget
    ]

    # Get all atoms for nearest token center atoms
    atom_array.set_annotation(
        "crop_mask",
        np.isin(
            atom_array.token_id,
            token_center_atoms[nearest_token_center_atom_ids].token_id,
        ),
    )


def sample_crop_strategy(crop_weights: dict[str, float]) -> str:
    """Samples cropping strategy with dataset-specific weights.

    Args:
        crop_weights (dict[str, float]):
            Dictionary of crop weights.

    Returns:
        str:
            Sampled cropping strategy.
    """
    crop_keys = np.array(list(crop_weights.keys()))
    crop_weights = np.array(list(crop_weights.values()))
    crop_weights /= crop_weights.sum()  # norm to 1

    return np.random.choice(crop_keys, p=crop_weights)


@log_runtime_memory(runtime_dict_key="runtime-target-structure-proc-crop")
def sample_crop_strategy_and_set_mask(
    atom_array: AtomArray,
    token_budget: int,
    crop_weights: dict[str, float],
    preferred_chain_or_interface: str | list[str, str] | None,
) -> str:
    """Samples cropping strategy and sets the crop mask.

    Running this function on an AtomArray will add the 'crop_mask' annotation in-place,
    which is True for atoms inside the crop and False for atoms outside the crop. Note
    that this actually does not yet crop the AtomArray itself, due to the downstream
    symmetry-expansion code requiring access to the full ground-truth structure.

    Args:
        atom_array (AtomArray):
            AtomArray of the input assembly to crop.
        token_budget (int):
            Number of tokens to sample.
        crop_weights (dict):
            Sampling weights of different crop strategies.
        preferred_chain_or_interface (str | list[str, str] | None):
            Optional chain ID or chain ID pair indicating the preferred chain or
            interface, from which reference atoms are selected. Generated by eq. 1 in
            AF3 SI for the weighted PDB dataset.
    Returns:
        str:
            Name of the sampled cropping strategy. Returns 'whole' if the whole assembly
            fits into the token budget. Should be one of ['contiguous', 'spatial',
            'spatial_interface', 'whole'].
    """

    # Take whole assembly if it fits in the budget as-is
    fits_in_budget = np.unique(atom_array.token_id).shape[0] <= token_budget

    if fits_in_budget:
        crop_strategy = "whole"

    # Otherwise use one of the different crop strategies to set a crop mask
    else:
        crop_strategy = sample_crop_strategy(crop_weights)

    if crop_strategy == "whole":
        set_whole_crop_mask(atom_array)
    elif crop_strategy == "contiguous":
        set_contiguous_crop_mask(atom_array=atom_array, token_budget=token_budget)
    elif crop_strategy == "spatial":
        set_spatial_crop_mask(
            atom_array=atom_array,
            token_budget=token_budget,
            preferred_chain_or_interface=preferred_chain_or_interface,
        )
    elif crop_strategy == "spatial_interface":
        set_spatial_interface_crop_mask(
            atom_array=atom_array,
            token_budget=token_budget,
            preferred_chain_or_interface=preferred_chain_or_interface,
        )

    return crop_strategy


class CroppingOutput(NamedTuple):
    """Output of the pre-cropping and crop mask setting.

    Attrs:
        atom_array (AtomArray):
            The (potentially precropped) AtomArray with the crop_mask annotation set.
        crop_strategy (str):
            The crop strategy that was applied. One of ['contiguous', 'spatial',
            'spatial_interface', 'whole'].
    """

    atom_array: AtomArray
    crop_strategy: str


def precrop_and_set_crop_mask(
    atom_array: AtomArray,
    apply_crop: bool,
    crop_config: dict,
    preferred_chain_or_interface: str | list[str, str] | None = None,
) -> CroppingOutput:
    """Applies chain-wise precropping (if enabled) and sets the main crop mask.

    This first applies chain-wise precropping (see precrop_chains and 2.5.4 of AF3 SI),
    and then samples and applies a standard cropping strategy (contiguous, spatial,
    spatial interface). Note that this second step of cropping only sets the 'crop_mask'
    annotation of the AtomArray (in-place), which is True for every atom in the crop and
    False for all others, and does not actually yet crop the AtomArray itself. This is
    due to downstream symmetry-expansion code requiring the full ground-truth.

    Args:
        atom_array (AtomArray):
            AtomArray of the input assembly to crop.
        apply_crop (bool):
            Whether to apply standard cropping.
        crop_config (dict):
            Configuration dictionary of crop settings. Refer to
            openfold3.projects.of3_all_atom.config.dataset_config_components.CropSettings
            for an overview of the expected keys.
        preferred_chain_or_interface (str | list[str, str]):
            Optional chain ID or chain ID pair indicating the preferred chain or
            interface, from which reference atoms for cropping are selected. Generated
            by eq. 1 in AF3 SI for the weighted PDB dataset.

    Returns:
        CroppingOutput:
            NamedTuple containing the (potentially precropped) AtomArray with the
            crop_mask annotation set, and the crop strategy that was applied.
    """
    precrop_settings = crop_config["precrop"]

    # 1) Apply chain-wise precropping if enabled
    if precrop_settings["enabled"]:
        atom_array = precrop_chains(
            atom_array=atom_array,
            preferred_chain_or_interface=preferred_chain_or_interface,
            n_chains=precrop_settings["n_chains"],
            interface_distance_threshold=precrop_settings[
                "interface_distance_threshold"
            ],
            ignore_ligands_below=precrop_settings["ignore_ligands_below"],
            ligand_inclusion_distance=precrop_settings["ligand_inclusion_distance"],
        )

    # 2) Sample and apply "standard" cropping strategy if enabled, setting the crop_mask
    #    attribute of the AtomArray
    if apply_crop:
        crop_strategy = sample_crop_strategy_and_set_mask(
            atom_array=atom_array,
            token_budget=crop_config["token_budget"],
            crop_weights=crop_config["crop_weights"],
            preferred_chain_or_interface=preferred_chain_or_interface,
        )
    else:
        crop_strategy = "whole"
        set_whole_crop_mask(atom_array)

    return CroppingOutput(
        atom_array=atom_array,
        crop_strategy=crop_strategy,
    )
