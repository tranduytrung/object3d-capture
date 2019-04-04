import os

os.environ["PANDA_PRC_DIR"] = os.path.dirname(os.path.realpath(__file__))
os.environ["PANDA_PRC_PATH"] = os.path.dirname(os.path.realpath(__file__))

from threading import Lock
from direct.showbase.ShowBase import ShowBase
from panda3d.core import VBase4, VBase3, VBase2
from panda3d.core import Filename, loadPrcFileData
from panda3d.core import DirectionalLight, AmbientLight, AntialiasAttrib
import numpy as np
import PIL.Image

class Panda3DRenderer:
    __lock = Lock()

    def __init__(self, cad_file, output_size=(512, 512), light_on=True, cast_shadow=True):
        # acquire lock since showbase cannot be created twice
        Panda3DRenderer.__lock.acquire()

        # set output size and init the base
        loadPrcFileData('', f'win-size {output_size[0]} {output_size[1]}') 
        base = ShowBase(windowType='offscreen')

        # coordinate for normalized and centered object
        nom_node = base.render.attach_new_node('normalized_obj')
        # Convert that to panda's unix-style notation.
        filepath = Filename.fromOsSpecific(cad_file).getFullpath()
        # load model and add to render root
        model_node = base.loader.load_model(filepath, noCache=True)
        model_node.reparent_to(nom_node)
        # scale to uinit sphere
        model_node_bounds = nom_node.getBounds()
        model_node.set_scale(1/model_node_bounds.radius)
        model_node_bounds = nom_node.getBounds()
        model_node.set_pos(-model_node_bounds.center)

        # directional light
        dlight = DirectionalLight('dlight')
        dlight.setColor(VBase4(0.15, 0.15, 0.15, 1))
        dlnp = base.render.attachNewNode(dlight)
        base.render.setLight(dlnp)
        # cast shadow from light
        # set auto shader
        if cast_shadow:
            model_node.setDepthOffset(1)
            dlight.setShadowCaster(True, 1024, 1024)
            lens = dlight.getLens()
            lens.set_near_far(0.866, 1000)
            lens.set_fov(1, 1)
            # lens.set_film_size(base.camLens.get_film_size())
            # lens.set_focal_length(base.camLens.get_focal_length())
            base.render.setShaderAuto()

        # ambient light
        alight = AmbientLight('alight')
        alight.setColor(VBase4(0.75, 0.75, 0.75, 1))
        alnp = base.render.attachNewNode(alight)
        base.render.setLight(alnp)

        # anti-alias
        base.render.setAntialias(AntialiasAttrib.MMultisample, 8)
        # init camera position
        self.coverage = 0.5
        # the default clear color
        self.clear_color = (0.0 ,0.0 , 0.0, 0.0)
        # translate in rendered image
        self.obj_translate = (0, 0)
        # light location (use location for cast shadow)
        self.dlight_params = {
            'theta': np.random.uniform(0, np.pi),
            'phi': np.random.uniform(0, 2*np.pi)
        }
        # object rotation
        self.obj_hpr = (0, 0, 0)

        self.base = base
        self.dlight = dlnp
        self.alight = alnp
        self.obj = nom_node
        self.camera = base.camera

        if not light_on:
            base.render.set_light_off()

    @property
    def clear_color(self):
        b4 = self._clear_color
        return b4.get_x(), b4.get_y(), b4.get_z(), b4.get_w()

    @clear_color.setter
    def clear_color(self, value):
        self._clear_color = VBase4(*value)

    @property
    def obj_hpr(self):
        hpr = self._obj_hpr
        return hpr.get_x(), hpr.get_y(), hpr.get_z()

    @obj_hpr.setter
    def obj_hpr(self, value):
        self._obj_hpr = VBase3(*value)
        
    @property
    def obj_translate(self):
        xy = self._obj_translate
        return xy.get_x(), xy.get_y()

    @obj_translate.setter
    def obj_translate(self, value):
        self._obj_translate = VBase2(*value)
        
    @property
    def obj_color(self):
        b4 = self.obj.get_color()
        return b4.get_x(), b4.get_y(), b4.get_z(), b4.get_w()

    @obj_color.setter
    def obj_color(self, value):
        self.obj.set_color(*value)

    def get_camera_radius(self):
        r = 1
        c = self.coverage
        lens = self.base.camLens
        tan_cam = lens.get_film_size()[1] / 2 / lens.get_focal_length()
        # tan_cam = lens.get_vfov() / 2 / lens.get_near()
        return r/(c*tan_cam)

    def get_dlight_pos(self):
        theta = self.dlight_params['theta']
        phi = self.dlight_params['phi']
        radius = self.get_camera_radius()
        x = radius*np.sin(theta)*np.cos(phi)
        y = radius*np.sin(theta)*np.sin(phi)
        z = np.absolute(radius*np.cos(theta))
    
        return VBase3(x, y, z)

    def render(self, binary=True):
        base = self.base
        # context
        base.win.setClearColor(self._clear_color)
        base.camera.set_pos(VBase3(0, -self.get_camera_radius(), 0))
        self.dlight.set_pos(self.get_dlight_pos())
        self.dlight.look_at(self.obj)
        self.obj.set_hpr(self._obj_hpr)

        # redner
        base.graphics_engine.render_frame()
        tex = base.win.get_screenshot()

        # export
        bytes_image = bytes(tex.get_ram_image_as('RGBA'))
        pil_image = PIL.Image.frombytes('RGBA', base.get_size(), bytes_image)
        pil_image = pil_image.transpose(PIL.Image.FLIP_TOP_BOTTOM)
        if tuple(self.target_translate) != (0, 0):
            np_clear_color = np.array(self._clear_color*255, dtype=int)
            translated_image = PIL.Image.new('RGBA', pil_image.size, tuple(np_clear_color))
            np_obj_translate = np.array(self.obj_translate, dtype=int)
            translated_image.paste(pil_image, tuple(np_obj_translate))
            pil_image = translated_image

        if not binary:
            return pil_image

        bin_image = PIL.Image.new('1', pil_image.size)
        bin_image.paste(1, None, pil_image)

        return pil_image, bin_image

    def random_context(self, light=True, obj_rotation=True, obj_color=True,
                       obj_translate=True, coverage=[0.1, 0.5]):
        r = 1  # unit sphere

        if coverage is not None:
            if hasattr(coverage, '__len__'):
                if len(coverage) == 2:
                    self.coverage = np.random.uniform(coverage[0], coverage[1])
                else:
                    self.coverage = coverage[0]
            elif isinstance(coverage, (int, float)):
                self.coverage = coverage
            else:
                raise ValueError()

        if obj_translate:
            self.target_translate = np.random.uniform(-0.5 + self.coverage/2, 0.5 - self.coverage/2, size=2) \
                                     * self.base.get_size()
            
        # light
        if light:
            self.dlight_params = {
                'theta': np.random.uniform(0, np.pi),
                'phi': np.random.uniform(0, 2*np.pi)
            }

        # obj rotate
        if obj_rotation:
            self.obj_hpr = np.random.uniform(
                -180, 180, size=3)

        if obj_color:
            self.obj_color = (*np.random.uniform(size=3), 1)
            
    def close(self):
        self.base.destroy()
        Panda3DRenderer.__lock.release()
        
    def __enter__(self):
        return self
            
    def __exit__(self, type, value, traceback):
        self.close()