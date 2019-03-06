import os
import object3d
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Generate random RGB images and their masks of a 3D Object")

    parser.add_argument(
        "--wavefront",
        required=True,
        metavar="/path/to/3d-object/",
        help="path to frontwave file (.obj)",
    )

    parser.add_argument(
        "--texture",
        required=True,
        metavar="/path/to/texture/",
        help="path to texture file (.jpg)",
    )

    parser.add_argument(
        "--out",
        required=True,
        metavar="/path/to/output/directory/",
        help="directory of the output",
    )

    parser.add_argument(
        "--num",
        required=False,
        default=100,
        type=int,
        help="number of output files",
    )

    parser.add_argument(
        "--width",
        required=False,
        default=512,
        type=int,
        help="width of output image",
    )

    parser.add_argument(
        "--height",
        required=False,
        default=512,
        type=int,
        help="height of output image",
    )

    parser.add_argument(
        "--prefix",
        required=False,
        default='',
        help="prefix of output file (prefix)001.rgb.png",
    )

    args = parser.parse_args()

    wf_file = args.wavefront
    tt_file = args.texture
    out_dir = args.out
    prefix = args.prefix
    width = args.width
    height = args.height
    num = args.num

    if not prefix:
        prefix = os.path.basename(wf_file).split(".")[0]

    obj3d = object3d.Object3DCapture(wf_file, tt_file, output_size=(width, height))
    
    digit_num = len(str((num)))
    for i in range(num):
        image_id = '{prefix}{it:0{width}}'.format(prefix=prefix, it=i, width=digit_num)
        fn = os.path.join(out_dir, image_id +'.rgb.png')
        fnb = os.path.join(out_dir, image_id + '.01.1.png')
        obj3d.random_context()
        obj3d.render_image().save(fn)
        obj3d.render_bimage().save(fnb)
        print(f"saved {image_id}")

if __name__ == "__main__":
    main()
