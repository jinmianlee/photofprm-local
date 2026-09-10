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
