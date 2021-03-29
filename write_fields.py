# -*- coding: utf-8 -*-

import numpy as np
import stk
from stk import StkMesh, Parallel, StkState, StkSelector, StkRank
import pickle
import pandas as pd
import scipy.spatial.qhull as qhull
from scipy.interpolate import LinearNDInterpolator


def p0_printer(par):
    iproc = par.rank

    def printer(*args, **kwargs):
        if iproc == 0:
            print(*args, **kwargs)

    return printer


par = Parallel.initialize()
printer = p0_printer(par)
printer(f"MPI: rank = {par.rank}; size = {par.size}")

# CDP, fix the bounds
cdp = pickle.load(open("cdpAMS.p", "rb"), fix_imports=True, encoding="latin1")
cdp.loc[cdp.x < 5e-9, ["x"]] = 0
cdp.loc[cdp.x > 2 * np.pi - 5e-9, ["x"]] = 2 * np.pi

# Create a STK mesh instance that holds MetaData and BulkData
mesh = StkMesh(par, ndim=3)

mesh.read_mesh_meta_data(
    "PeriodicChannel-SWH-ndtw.exo",
    purpose=stk.DatabasePurpose.READ_MESH,
    auto_decomp=True,
    auto_decomp_type="rcb",
    auto_declare_fields=True,
)

# Register fields
velocity = mesh.meta.declare_vector_field("velocity", number_of_states=3)
avg_velocity = mesh.meta.declare_vector_field("average_velocity", number_of_states=3)
turb_ke = mesh.meta.declare_scalar_field("turbulent_ke")
turb_visc = mesh.meta.declare_scalar_field("turbulent_viscosity")
avg_prod = mesh.meta.declare_scalar_field("average_production")
avg_rk = mesh.meta.declare_scalar_field("avg_res_adequacy_parameter")
res_adeq = mesh.meta.declare_scalar_field("resolution_adequacy_parameter")
k_ratio = mesh.meta.declare_scalar_field("k_ratio")
avg_tke_res = mesh.meta.declare_scalar_field("average_tke_resolved")

# Register the fields on desired parts
velocity.add_to_part(
    mesh.meta.universal_part,
    mesh.meta.spatial_dimension,
    init_value=np.array([10.0, 0.0, 0.0]),
)
avg_velocity.add_to_part(
    mesh.meta.universal_part,
    mesh.meta.spatial_dimension,
    init_value=np.array([10.0, 0.0, 0.0]),
)
turb_ke.add_to_part(mesh.meta.universal_part, init_value=np.array([0.0]))
turb_visc.add_to_part(mesh.meta.universal_part, init_value=np.array([0.0]))
avg_prod.add_to_part(mesh.meta.universal_part, init_value=np.array([0.0]))
avg_rk.add_to_part(mesh.meta.universal_part, init_value=np.array([0.0]))
res_adeq.add_to_part(mesh.meta.universal_part, init_value=np.array([0.0]))
k_ratio.add_to_part(mesh.meta.universal_part, init_value=np.array([0.0]))
avg_tke_res.add_to_part(mesh.meta.universal_part, init_value=np.array([0.0]))

# Commit the metadata and load the mesh. Also create edges at this point (default is False)
mesh.populate_bulk_data(create_edges=True)
printer("Metadata is committed: ", mesh.meta.is_committed)

mesh.stkio.read_defined_input_fields(0.0)

# Access the coordinate field
coords = mesh.meta.coordinate_field
printer("Coordinates field: ", coords.name)

# Access a part by name
part = mesh.meta.get_part("unspecified-2-hex")
printer(f"Part exists = {(not part.is_null)}")

# Get a stk::mesh::Selector instance for all locally-owned entities of this part
sel = part & (mesh.meta.locally_owned_part | mesh.meta.globally_shared_part)

# Check if the selector is empty for a particular entity type
print(f"Fluid_part has elems: {not sel.is_empty(StkRank.ELEM_RANK)}")

# Update fields
buf = 0.2
for k, bkt in enumerate(mesh.iter_buckets(sel, StkRank.NODE_RANK)):

    xyz = coords.bkt_view(bkt)
    xmin, xmax = np.min(xyz[:, 0]) - buf, np.max(xyz[:, 0]) + buf
    ymin, ymax = np.min(xyz[:, 1]) - buf, np.max(xyz[:, 1]) + buf
    zmin, zmax = np.min(xyz[:, 2]) - buf, np.max(xyz[:, 2]) + buf
    xyz_cdp = cdp.loc[
        (xmin < cdp.x)
        & (cdp.x < xmax)
        & (ymin < cdp.y)
        & (cdp.y < ymax)
        & (zmin < cdp.z)
        & (cdp.z < zmax),
        ["x", "y", "z"],
    ]
    idx = xyz_cdp.index
    tri = qhull.Delaunay(xyz_cdp)

    fields = {
        "ux": {"arr": velocity.bkt_view(bkt), "comp": 0},
        "uy": {"arr": velocity.bkt_view(bkt), "comp": 1},
        "uz": {"arr": velocity.bkt_view(bkt), "comp": 2},
        "avgUx": {"arr": avg_velocity.bkt_view(bkt), "comp": 0},
        "avgUy": {"arr": avg_velocity.bkt_view(bkt), "comp": 1},
        "avgUz": {"arr": avg_velocity.bkt_view(bkt), "comp": 2},
        "tke": {"arr": turb_ke.bkt_view(bkt)},
        "tvisc": {"arr": turb_visc.bkt_view(bkt)},
        "avgProd": {"arr": avg_prod.bkt_view(bkt)},
        "avgRk": {"arr": avg_rk.bkt_view(bkt)},
        "rk": {"arr": res_adeq.bkt_view(bkt)},
        "alpha": {"arr": k_ratio.bkt_view(bkt)},
        "kres": {"arr": avg_tke_res.bkt_view(bkt)},
    }
    lcdp = cdp.loc[idx]
    for i, (key, value) in enumerate(fields.items()):
        interpolator = LinearNDInterpolator(tri, lcdp[key].to_numpy())
        fl = interpolator(xyz)
        if "comp" in value.keys():
            value["arr"][:, value["comp"]] = fl
        else:
            value["arr"][:] = fl

purpose = stk.DatabasePurpose.WRITE_RESULTS
stkio = mesh.stkio
fh = stkio.create_output_mesh("./out/output.e", purpose=purpose)
for fld in [
    "velocity",
    "average_velocity",
    "turbulent_ke",
    "turbulent_viscosity",
    "average_production",
    "avg_res_adequacy_parameter",
    "resolution_adequacy_parameter",
    "k_ratio",
    "average_tke_resolved",
    "ndtw",
]:
    f = mesh.meta.get_field_by_name(fld)
    stkio.add_field(fh, f)

stkio.begin_output_step(fh, 0.0)
stkio.write_defined_output_fields(fh)
stkio.end_output_step(fh)

del mesh
par.finalize()
