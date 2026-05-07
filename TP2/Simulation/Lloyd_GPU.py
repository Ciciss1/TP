import time
import numpy as np
import torch
from scipy.spatial import Voronoi

def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        print(f"Using GPU: {props.name} | "
                f"Memory: {props.total_memory / 1e9:.2f} GB | "
                f"SM : {props.multi_processor_count}")
        return dev
    print("Using CPU")
    return torch.device("cpu")

def pack_voronoi_regions(
        vor: Voronoi,
        generators: np.ndarray,
        free_idx: np.ndarray,
        N: int,
        L: float
    ):
    '''
    Pack the Voronoi regions into a tensor for GPU processing
    Inputs:
        vor : Voronoi object from scipy.spatial
        generators : array of generator positions
        free_idx : indices of the free generators
        N : number of generators
        L : physical size of the box
    Outputs:
        flat_x, flat_y : coordinates of the vertices
        flat_id : index of the generator for each vertex
    '''
    center_offset = 4 * N
    vertices = vor.vertices
    point_region = vor.point_region

    flat_x_list = []
    flat_y_list = []
    flat_id_list = []

    for idx in free_idx:
        region = vor.regions[point_region[idx + center_offset]]

        if -1 in region or len(region) < 3:
            continue

        verts = vertices[region]
        ref = generators[idx]

        dv = verts - ref
        dv = dv - L * np.round(dv / L)
        verts_c = ref + dv

        n = len(verts_c)
        flat_x_list.append(verts_c[:, 0])
        flat_y_list.append(verts_c[:, 1])
        flat_id_list.append(np.full(n, idx, dtype=np.int64))

    if not flat_x_list:
        return None, None, None
        
    return np.concatenate(flat_x_list), np.concatenate(flat_y_list), np.concatenate(flat_id_list)

def compute_centroid(
        flat_x: np.ndarray,
        flat_y: np.ndarray,
        flat_id: np.ndarray,
        pos_gpu: torch.Tensor,
        N: int,
        L: float,
        device: torch.device
    ) -> torch.Tensor:
    '''
    Compute the centroids of the Voronoi cells
    Inputs:
        flat_x : numpy array of shape (M,) containing the x-coordinates of the vertices
        flat_y : numpy array of shape (M,) containing the y-coordinates of the vertices
        flat_id : numpy array of shape (M,) containing the index of the closest generator for each vertex
        pos_gpu : tensor of shape (N, 2) containing the coordinates of the generators
        N : number of generators
        L : physical size of the box
        device : torch device to use for computation
    Outputs:
        centroids : tensor of shape (N, 2) containing the coordinates of the centroids
    '''
    x = torch.tensor(flat_x, dtype=torch.float64, device=device)
    y = torch.tensor(flat_y, dtype=torch.float64, device=device)
    idx = torch.tensor(flat_id, dtype=torch.int64, device=device)
    K = len(idx)

    is_last = torch.cat([idx[1:] != idx[:-1], torch.tensor([True], device=device)])
    is_first = torch.cat([torch.tensor([True], device=device), idx[:-1] != idx[1:]])
    first_idx = torch.where(is_first)[0]

    next_idx = torch.arange(K, dtype=torch.int64, device=device)
    next_idx[~is_last] += 1
    next_idx[torch.where(is_last)[0]] = first_idx

    xn = x[next_idx]
    yn = y[next_idx]

    cross = x * yn - xn * y

    sum_cross = torch.zeros(N, dtype=torch.float64, device=device)
    sum_cx = torch.zeros(N, dtype=torch.float64, device=device)
    sum_cy = torch.zeros(N, dtype=torch.float64, device=device)

    sum_cross.scatter_add_(0, idx, cross)
    sum_cx.scatter_add_(0, idx, (x + xn) * cross)
    sum_cy.scatter_add_(0, idx, (y + yn) * cross)

    valid = sum_cross.abs() > 1e-10

    ref_x = pos_gpu[:, 0]
    ref_y = pos_gpu[:, 1]

    denom = torch.where(valid, 3 * sum_cross, torch.ones(N, dtype=torch.float64, device=device))

    cx = torch.where(valid, sum_cx / denom, ref_x) % L
    cy = torch.where(valid, sum_cy / denom, ref_y) % L

    return torch.stack([cx, cy], dim=1)

def lloyd_hybrid(
        generators: np.ndarray,
        boundary_mask: np.ndarray,
        L: float,
        n_iter: int = 50,
        tol: float = 0.5,
        device: torch.device = None
    ) -> np.ndarray:
    '''
    Lloyd's algorithm with Voronoi scipy for CPU + centroid computation on GPU
    Inputs:
        generators : numpy array of shape (N, 2) containing the coordinates of the generators
        boundary_mask : numpy array of shape (R, R) containing a boolean mask of the valid region
        L : physical size of the box
        n_iter : maximum number of iterations
        tol : tolerance for convergence
        device : torch device to use for computation (if None, will auto-detect)
    Outputs:
        relaxed_generators
    '''
    if device is None:
        device = get_device()

    N = len(generators)
    free_idx = np.where(boundary_mask)[0]
    F = len(free_idx)

    if F == 0:
        return generators.copy()
    
    pos_cpu = generators.astype(np.float64).copy()
    pos_gpu = torch.tensor(pos_cpu, dtype=torch.float64, device=device)
    free_t = torch.tensor(free_idx, dtype=torch.int64, device=device)

    for it in range(n_iter):
        t0 = time.perf_counter()

        images = [pos_cpu + np.array([dx, dy])for dx in (-L, 0, L) for dy in (-L, 0, L)]
        all_gen = np.vstack(images)
        vor = Voronoi(all_gen)

        flat_x, flat_y, flat_id = pack_voronoi_regions(vor, pos_cpu, free_idx, N, L)

        if flat_x is None:
            break

        centroids = compute_centroid(flat_x, flat_y, flat_id, pos_gpu, N, L, device)

        new_pos = pos_gpu.clone()
        new_pos[free_t] = centroids[free_t]

        diff = new_pos[free_t] - pos_gpu[free_t]
        diff = diff - L * torch.round(diff / L)
        delta = diff.norm(dim=1).max().item()

        pos_gpu = new_pos
        pos_cpu = pos_gpu.cpu().numpy()

        if delta < tol:
            break

    return pos_cpu

class LloydHybrid:
    def relaxation_GPU(
        self,
        generators: np.ndarray,
        boundary_mask: np.ndarray,
        n_iter: int = 50,
        tol: float = 0.5,
        ) -> np.ndarray:
        '''
        Lloyd's algorithm with Voronoi scipy for CPU + centroid computation on GPU
        Inputs:
            generators : numpy array of shape (N, 2) containing the coordinates of the generators
            boundary_mask : numpy array of shape (R, R) containing a boolean mask of the valid region
            n_iter : maximum number of iterations
            tol : tolerance for convergence
        Outputs:
            relaxed_generators
        '''
        device = get_device()
        
        return lloyd_hybrid(generators, boundary_mask, self.L, n_iter, tol, device)