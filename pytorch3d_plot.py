import torch
from pytorch3d.io import load_obj

from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import plot_scene
from pytorch3d.renderer import TexturesVertex

def orthogonal_procrustes(M):
    u, s, v = M.svd()
    return u.mm(v.T), s.sum()

def align(A, M):
    R, s = orthogonal_procrustes(M)
    return A.mm(R) * s

vis_file = 'outputs/manogcnx3_1x_freihand_aligned+uvd/inference/freihand_test/visualize.pth'
results = torch.load(vis_file)

idxs = [1,2,3,4,5]
for idx in idxs:
    verts, faces = results[idx]['uvds'], results[idx]['faces']
    # Initialize each vertex to be white in color.
    verts_rgb = torch.ones_like(verts)[None] -0.35  # (1, V, 3)
    textures = TexturesVertex(verts_features=verts_rgb)

    mesh = Meshes(
        verts=[verts],   
        faces=[faces],
        textures=textures
    )

    # Render the plotly figure
    fig = plot_scene({
        "subplot1": {
            "cow_mesh": mesh
        }
    })

    fig.write_html(f'{idx}.html')
