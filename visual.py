import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from part3 import SFM


def seperate_by_color(candidates, colors):
    red_x = [point[0] for ind, point in enumerate(candidates) if
             colors[ind] is "red"]
    red_y = [point[1] for ind, point in enumerate(candidates) if
             colors[ind] is "red"]
    green_x = [point[0] for ind, point in enumerate(candidates) if
               colors[ind] is "green"]
    green_y = [point[1] for ind, point in enumerate(candidates) if
               colors[ind] is "green"]
    return green_x, green_y, red_x, red_y


def visualize(part1_candidates, part1_auxiliary, prev_container, current_container, focal, pp):
    fig, (part1, part2, part3) = plt.subplots(1, 3, figsize=(12, 6))
    fig.canvas.set_window_title('Mobileye Project 2020')
    img = Image.open("mobileye_logo.png")
    img = img.resize((280, 200), Image.ANTIALIAS)
    fig.figimage(img, fig.bbox.xmax - 300, fig.bbox.ymin - 15)
    plt.suptitle(current_container.img_path[current_container.img_path.rfind('\\') + 1:-4], size=16)
    part1.set_title('candidates')
    part1.imshow(current_container.img)
    red_x, red_y, green_x, green_y = seperate_by_color(part1_candidates, part1_auxiliary)
    part1.plot(red_x, red_y, 'ro', color='r', markersize=4)
    part1.plot(green_x, green_y, 'ro', color='g', markersize=4)

    part2.set_title('traffic lights')
    part2.imshow(current_container.img)
    red_x, red_y, green_x, green_y = seperate_by_color(current_container.traffic_light, current_container.auxiliary)
    part2.plot(red_x, red_y, 'ro', color='r', markersize=4)
    part2.plot(green_x, green_y, 'ro', color='g', markersize=4)

    part3.set_title('distances')
    if prev_container is not None:
        norm_prev_pts, norm_curr_pts, R, norm_foe, tZ = SFM.prepare_3D_data(prev_container,current_container,
                                                                            focal, pp)
        norm_rot_pts = SFM.rotate(norm_prev_pts, R)
        rot_pts = SFM.unnormalize(norm_rot_pts, focal, pp)
        foe = np.squeeze(SFM.unnormalize(np.array([norm_foe]), focal, pp))
        part3.imshow(current_container.img)
        curr_p = current_container.traffic_light
        part3.plot(curr_p[:][0], curr_p[:][1], 'b+')

        for i in range(len(curr_p)):
            part3.plot([curr_p[i][0], foe[0]], [curr_p[i][1], foe[1]], 'b')
            if current_container.valid[i]:
                part3.text(curr_p[i][0], curr_p[i][1],
                           r'{0:.1f}'.format(current_container.traffic_lights_3d_location[i, 2]), color='r')
        part3.plot(foe[0], foe[1], 'r+')
        part3.plot(rot_pts[:, 0], rot_pts[:, 1], 'g+')

    plt.show()
