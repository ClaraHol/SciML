from dataclasses import dataclass
import torch
from dataclasses import dataclass
from scipy.io import loadmat, savemat
from pathlib import Path

@dataclass
class Book_data:
    """
    Load data from the book. 
    """
    mat: torch.Tensor

    @classmethod
    def exctract_mat_data(cls, header: str) -> None:
        "Extract data from CYLINDER_ALL"
        data = loadmat("data/raw/CYLINDER_ALL.mat")
        assert header in data.keys(), "Header not found"
        
        mat = torch.Tensor(data[header])

        torch.save(mat, "data/processed/" +header + ".pt")

    
    
    @classmethod
    def load_data(cls, header: str) -> 'Book_data':
        "Load tensor data from data/processed"
        data_path = Path("data/processed/" + header + ".pt")
        
        if data_path.exists():
            mat = torch.load(data_path)
        else:
            cls.exctract_mat_data(header)
            mat = torch.load(data_path)
        
        return cls(mat = mat)
        
    
            

