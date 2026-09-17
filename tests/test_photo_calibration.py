import numpy as np
import trimesh
import pytest
from backend.print_colors import project_flat_colors
from backend.app import Options


def test_calibration_places_two_small_marks_without_painting_the_back():
    mesh=trimesh.creation.icosphere(subdivisions=4)
    photo=np.full((400,400,4),255,dtype=np.uint8);photo[...,:3]=[202,156,91]
    photo[125:155,95:125,:3]=[15,15,15]
    photo[125:155,275:305,:3]=[15,15,15]
    photo[240:262,189:211,:3]=[120,62,60]
    landmarks=[{'model':[-.45,.3,.84],'photo':[.275,.35]},
               {'model':[.45,.3,.84],'photo':[.725,.35]},
               {'model':[0.,-.1,.995],'photo':[.5,.625]}]
    provenance=project_flat_colors(mesh,photo,4,0,0,landmarks=landmarks)
    centers=mesh.triangles_center;rgb=np.asarray(mesh.visual.face_colors)[:,:3]
    assert provenance['alignment_method']=='paired_landmarks_affine'
    assert np.any((centers[:,2]>.65)&(rgb.max(axis=1)<60))
    assert np.all(rgb[centers[:,2]<-.5].max(axis=1)>100)


def test_bad_pair_geometry_is_rejected_and_request_points_are_bounded():
    mesh=trimesh.creation.icosphere(subdivisions=2)
    photo=np.full((200,200,4),255,dtype=np.uint8)
    collinear=[{'model':[x,0,.8],'photo':[.5+x/2,.5]} for x in [-.4,0,.4]]
    with pytest.raises(ValueError,match='共线'):
        project_flat_colors(mesh,photo,2,0,0,landmarks=collinear)
    with pytest.raises(ValueError):Options(photo_landmarks=collinear[:2])
    with pytest.raises(ValueError):Options(shape_steps=100)
