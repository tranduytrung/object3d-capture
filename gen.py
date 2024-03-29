import os
import object3d
import argparse
import numpy
import glob
import PIL.Image

# python gen.py --wavefront=robot.obj --texture=robot.jpg --background=D:\tranduytrung\background\**\*.jpg --num=30 --out=D:\tranduytrung\auto-generated\val

def augment_random_background(image, bg_paths, mask):
    bg_id = numpy.random.randint(0, len(bg_paths))
    bg_image = PIL.Image.open(bg_paths[bg_id])
    # resize
    bg_size = bg_image.size
    desired_size = image.size[0] # square image
    ratio = float(desired_size)/min(bg_size)
    new_size = tuple([int(x*ratio) for x in bg_size])
    bg_image = bg_image.resize(new_size, PIL.Image.BILINEAR)
    # random crop
    left = 0 if new_size[0] == desired_size else numpy.random.randint(0, new_size[0] - desired_size)
    top = 0 if new_size[1] == desired_size else numpy.random.randint(0, new_size[1] - desired_size)
    bg_image = bg_image.crop((left, top, left + desired_size, top + desired_size))

    # convert RGB if background is not
    if bg_image.mode != 'RGB':
        bg_image = bg_image.convert('RGB')

    # convert mask to greyscale since PIL does not work corectly on mode 1?
    # mask = mask.convert(mode='L')

    # paste to background image
    bg_image.paste(image, None, mask)
    return bg_image

def generate2(wf_file, out_dir, bg_paths, width=512, height=512, random_color=False, cast_shadow=True, light_on=True,
        num_instances=100, coverage=(0.1, 0.5), prefix=None, class_id=None, log=None):
    if not prefix:
        prefix = os.path.basename(wf_file).split(".")[0]

    if not class_id:
        class_id = '0'

    fn, ext = os.path.splitext(wf_file)
    if ext == '.obj':
        egg_file = fn + '.egg'
        if not os.path.isfile(egg_file):
            from obj2egg import obj2egg
            log('converting to egg file')
            obj2egg(wf_file, egg_file)
        wf_file = egg_file

    with object3d.Panda3DRenderer(wf_file, output_size=(width, height), 
                cast_shadow=cast_shadow, light_on=light_on) as obj3d:
        digit_num = len(str(num_instances))
        for i in range(num_instances):
            image_id = '{prefix}{it:0{width}}'.format(prefix=prefix, it=i, width=digit_num)
            fn = os.path.join(out_dir, image_id +'.rgb.png')
            fnb = os.path.join(out_dir, image_id + f'.{class_id}.1.png')
            obj3d.random_context(obj_color=random_color, coverage=coverage)
            image, mask = obj3d.render()
            image = image.convert('RGB')
            
            if bg_paths:
                image = augment_random_background(image, bg_paths, mask)
            image.save(fn)
            mask.save(fnb)
            if log:
                log(f"saved {fn}")

def generate(wf_file, tt_file, mtl_file, out_dir, bg_paths, width=512, height=512, 
        num_instances=100, coverage=(0.1, 0.5), prefix=None, class_id=None, log=None):
    if not prefix:
        prefix = os.path.basename(wf_file).split(".")[0]

    if not class_id:
        class_id = '0'

    obj3d = object3d.Object3DCapture(wf_file, tt_file, mtl_file, output_size=(width, height))
    
    digit_num = len(str(num_instances))
    for i in range(num_instances):
        image_id = '{prefix}{it:0{width}}'.format(prefix=prefix, it=i, width=digit_num)
        fn = os.path.join(out_dir, image_id +'.rgb.png')
        fnb = os.path.join(out_dir, image_id + f'.{class_id}.1.png')
        obj3d.random_context(coverage=coverage)
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
        "-i",
        "--obj",
        required=True,
        metavar="/path/to/3d-object/",
        help="path to 3d object file (.obj)",
    )

    parser.add_argument(
        "-t",
        "--texture",
        required=False,
        default=None,
        metavar="/path/to/texture/",
        help="path to texture file (.jpg)",
    )

    parser.add_argument(
        "-m",
        "--mtl",
        required=False,
        default=None,
        metavar="/path/to/material/",
        help="path to texture file (.mtl)",
    )

    parser.add_argument(
        "-c",
        "--coverage",
        required=False,
        default=(0.1, 0.5),
        nargs='+',
        type=float,
        metavar="default to 0.1 -> 0.5",
        help="range of coverage",
    )

    parser.add_argument(
        "-o",
        "--out",
        required=True,
        metavar="/path/to/output/directory/",
        help="directory of the output",
    )

    parser.add_argument(
        "-n",
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
        "-b",
        "--background",
        required=False,
        default='',
        help="glob of background. Ex: ./*.jpg",
    )

    args = parser.parse_args()

    wf_file = args.obj
    tt_file = args.texture
    mtl_file = args.mtl
    out_dir = args.out
    prefix = args.prefix
    width = args.width
    height = args.height
    num = args.num
    coverage = args.coverage
    bg_paths = glob.glob(args.background, recursive=True) if args.background else None

    generate2(wf_file, out_dir, bg_paths, width=width, height=height, num_instances=num, prefix=prefix, log=print, coverage=coverage)

if __name__ == "__main__":
    main()
