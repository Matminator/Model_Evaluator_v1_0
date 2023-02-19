# Standard imports:
import numpy as np
import pandas as pd

# ASE imports
import ase
from ase.io.trajectory import Trajectory

# NequIP
import nequip
from nequip.ase import NequIPCalculator

def extract_all_data(data, new_calculator = None):
    """
    This function either extracts energies and forces - or computes all 
    these if given a new calculator. Thereafter, stores these along with
    the atomic positions, force magnitude, atomic species and give each
    structure (atoms objece) an uniqe label (0,1,2 ...).

    INPUT: 
    data: ASE Trajectory
    new_calculator: ASE calculator (usually deep learning model in)

    Returns: pandas dataframe with above mentioned contents
    """

# *** Initializing all arrays for pd dataframe ***********

    energies = np.array([])
    
    forces = np.array([])
    positions = np.array([])
    
    species = np.array([]) # (atomic species)
    
    N_list = np.array([]) # List providing number atoms per species
    
    # To keep track of which structure (atoms object) the atom is part of:
    atoms_label = 0
    atoms_list = np.array([])
    
# *** Looping over all atoms objects form data file ***********

    for atoms in data:
        
        if(new_calculator != None): # Assigns new calculator, if provided
            atoms.calc = new_calculator
        
        # Getting atomic species:
        new_species = atoms.get_chemical_symbols()
        species = np.append(species, new_species)
        
        n = len(new_species) # number of atoms
        
        # Getting atoms forces and positions:
        new_forces = atoms.get_forces()
        new_positions = atoms.get_positions()
        
        # If no forces are stored in the forces array yet. (this is force
        # np.vstack to work - needs the correct dimentions):
        if(len(forces) > 0):
            forces = np.vstack((forces, new_forces))
            positions = np.vstack((positions, new_positions))
        else:
            forces = new_forces
            positions = new_positions
            
        # Getting atoms energy:
        new_energy = atoms.get_potential_energy()
        energies = np.append(energies,
                             np.ones(n) * new_energy)
        
        # Getting number of atoms -- atoms object:
        N_list = np.append(N_list,
                             np.ones(n) * n)
            
        # Species_list (all of current atoms are part of the same label/
        # structure):
        atoms_list = np.append(atoms_list,
                                 np.ones(n) * atoms_label)
        atoms_label += 1
    
    # Computing magnitude of force
    forces_magnitude = np.linalg.norm(forces, axis = 1)
    
# *** Constructing pandas dataframe for output ***********    

    df = pd.DataFrame(data = forces)
    df = df.rename(columns={0: "f_x",1: "f_y", 2: "f_z"})
    
    df['|f|'] = forces_magnitude
    
    df['pos_x'] = positions[:,0]
    df['pos_y'] = positions[:,1]
    df['pos_z'] = positions[:,2]
    
    df['species'] = species
    
    df['atoms_label'] = atoms_list
    
    df['E'] = energies
    
    df['N'] = N_list
    
    return df
