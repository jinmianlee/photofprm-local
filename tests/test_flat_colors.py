import numpy as np
from backend.print_colors import flat_palette


def test_material_color_survives_broad_shadow_without_erasing_black_and_white():
    picture = np.full((180, 300, 4), 255, dtype=np.uint8)
    shade = np.linspace(.3, 1., 300)
    picture[..., :3] = np.rint(np.array([225, 130, 42])[None, None, :]*shade[None, :, None])
    picture[25:70, 30:95, :3] = [12, 12, 12]
    picture[105:155, 190:270, :3] = [235, 235, 235]
    labels, palette = flat_palette(picture, 3)
    orange = labels[80:100]
    main = np.argmax(np.bincount(orange.ravel()))
    assert np.mean(orange == main) > .95
    black = labels[35:60, 40:85]
    white = labels[115:145, 200:260]
    assert np.mean(black != main) > .95 and np.mean(white != main) > .95
    assert np.median(black) != np.median(white)
    assert len(palette) == 3


def test_small_eyes_nose_and_stripes_survive_a_large_warm_foreground():
    photo=np.full((400,400,4),255,dtype=np.uint8)
    photo[...,:3]=[198,153,94]
    for x in [115,250]:
        photo[130:150,x:x+20,:3]=[16,19,11]
    photo[205:225,175:210,:3]=[126,67,62]
    photo[242:285,130:255,:3]=[218,204,174]
    photo[65:85,160:166,:3]=[145,113,78]
    photo[65:85,180:186,:3]=[145,113,78]
    labels,palette=flat_palette(photo,5)
    eye=int(labels[139,123]);nose=int(labels[215,185]);fur=int(labels[320,200])
    assert eye!=fur and nose!=fur and eye!=nose
    assert palette[eye].max()<60
    assert labels[73,163]!=fur and labels[73,183]!=fur


def test_completely_dark_subject_does_not_require_an_empty_regular_cluster():
    photo=np.full((80,100,4),255,dtype=np.uint8)
    photo[...,:3]=[13,17,12]
    photo[20:60,30:70,:3]=[22,24,20]
    labels,palette=flat_palette(photo,5)
    assert labels.shape==photo.shape[:2]
    assert palette.max()<80
