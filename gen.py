import os
import object3d
import argparse
import numpy
import glob
import PIL.Image

# python gen.py --wavefront=robot.obj --texture=robot.jpg --background=D:\tranduytrung\background\**\*.jpg --num=30 --out=D:\tranduytrung\auto-generated\val

def augment_random_background(image, bg_paths, mask):
    bg_id = numpy.random.randint(0, len(bg_paths))
    bg_image = PIL.Image.open(bg_paths[bg_id]).resize(image.size, PIL.Image.BILINEAR)
    # convert RGB if background is not
    if bg_image.mode != 'RGB':
        bg_image = bg_image.convert('RGB')

    # convert mask to greyscale since PIL does not work corectly on mode 1?
    # mask = mask.convert(mode='L')

    # paste to background image
    bg_image.paste(image, None, mask)
    return bg_image

def generate(wf_file, tt_file, out_dir, bg_paths, width=512, height=512, size=100, prefix=None, class_id=None, log=None):
    if not prefix:
        prefix = os.path.basename(wf_file).split(".")[0]

    if not class_id:
        class_id = '0'

    obj3d = object3d.Object3DCapture(wf_file, tt_file, output_size=(width, height))
    
    digit_num = len(str(size))
    for i in range(size):
        image_id = '{prefix}{it:0{width}}'.format(prefix=prefix, it=i, width=digit_num)
        fn = os.path.join(out_dir, image_id +'.rgb.png')
        fnb = os.path.join(out_dir, image_id + f'.{class_id}.1.png')
        obj3d.random_context()
        image = obj3d.render_image()
        mask = obj3d.render_bimage()
        if bg_paths:
            image = augment_random_background(image, bg_paths, mask)
        image.save(fn)
        mask.save(fnb)
        if log:
            log(f"saved {image_id}")

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

    parser.add_argument(
        "--background",
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
    bg_paths = glob.glob(args.background, recursive=True) if args.background else None

    generate(wf_file, tt_file, out_dir, bg_paths, width=width, height=height, size=num, prefix=prefix, log=print)

if __name__ == "__main__":
    main()
