""" Capture 2D images of a 3D object in different point of views
"""
import moderngl
import numpy as np
from objloader import Obj
from PIL import Image, ImageOps
from pyrr import Matrix44, Vector4

class Object3DCapture:
    """ Capture the 2D images of an 3D Object"""
    def __init__(self, wavefront_path, texture_path, output_size):
        # context and shader program
        self.ctx = ctx = moderngl.create_standalone_context()
        ctx.enable(moderngl.DEPTH_TEST)
        prog = ctx.program(
            vertex_shader='''
                #version 330
                uniform mat4 Mvp;
                in vec3 in_vert;
                in vec3 in_norm;
                in vec2 in_text;
                out vec3 v_vert;
                out vec3 v_norm;
                out vec2 v_text;
                void main() {
                    v_vert = in_vert;
                    v_norm = in_norm;
                    v_text = in_text;
                    gl_Position = Mvp * vec4(v_vert, 1.0);
                }
            ''',


            fragment_shader='''
                #version 330
                uniform sampler2D Texture;
                uniform vec4 Color;
                uniform vec3 Light;
                in vec3 v_vert;
                in vec3 v_norm;
                in vec2 v_text;
                out vec4 f_color;
                void main() {
                    float lum = dot(normalize(v_norm), normalize(v_vert - Light));
                    lum = acos(lum) / 3.14159265;
                    lum = clamp(lum, 0.0, 1.0);
                    lum = lum * lum;
                    lum = smoothstep(0.0, 1.0, lum);
                    lum *= smoothstep(0.0, 80.0, v_vert.z) * 0.3 + 0.7;
                    lum = lum * 0.5 + 0.5;
                    vec3 color = texture(Texture, v_text).rgb;
                    color = color * (1.0 - Color.a) + Color.rgb * Color.a;
                    f_color = vec4(color * lum, 1.0);
                }
            ''',
        )
        
        # load wavefront obj file and the texture
        self.obj = obj = Obj.open(wavefront_path)
        img = Image.open(texture_path).convert('RGB').transpose(Image.FLIP_TOP_BOTTOM)

        self.light = prog['Light']
        self.color = prog['Color']
        self.mvp = prog['Mvp']

        texture = ctx.texture(img.size, 3, img.tobytes())
        texture.build_mipmaps()
        texture.use()

        vbo = ctx.buffer(obj.pack('vx vy vz nx ny nz tx ty'))
        self.vao = ctx.simple_vertex_array(prog, vbo, 'in_vert', 'in_norm', 'in_text')
        
        # editable parameters
        self.cam_angle = 45.0
        self.cam_pos_coe_min = 2.0
        self.cam_pos_coe_max = 10.0
        
        # setup default
        self.light_color = (1.0, 1.0, 1.0, 0.25)
        self.output_size = output_size
        self.random_context()
        
    @property
    def output_size(self):
        return self.fbo.size
    
    @output_size.setter
    def output_size(self, value):
        assert len(value) == 2
        w, h = value

        if hasattr(self, 'fbo'):
            self.fbo.release()
            
        self.fbo = self.ctx.simple_framebuffer((w, h))
        self.fbo.use()
    
    def render(self, clr_color=(1.0, 1.0, 1.0)):
        fbo = self.fbo
        fbo.clear(*clr_color)
        
        proj = Matrix44.perspective_projection(self.cam_angle, 1.0, 0.01, 1000.0)
        lookat = Matrix44.look_at(
            (*self.cam_pos,),
            (*self.target_pos,),
            (0.0, 1.0, 0.0),
        )

        self.light.value = tuple(self.light_pos)
        self.color.value = self.light_color
        self.mvp.write((proj * lookat).astype('f4').tobytes())

        self.vao.render()
        
        return fbo.read()
    
    def render_image(self, clr_color=(0.0, 0.0, 0.0)):
        return Image.frombytes('RGB', self.fbo.size, self.render(clr_color), 'raw')
    
    def render_bimage(self):
        t = self.light_color
        self.light_color = (0.0, 0.0, 0.0, 1.0)
        img = Image.frombytes('RGB', self.fbo.size, self.render((1.0, 1.0, 1.0)), 'raw')
        self.light_color = t
        return ImageOps.invert(img.convert('L', dither=None)).convert("1")
            
    def random_context(self):
        obj = self.obj
        
        cam_ratio = self.fbo.size[0] / self.fbo.size[1]
        center = np.mean(obj.vert, axis=0)
        r = np.sqrt(np.sum((np.max(obj.vert, axis=0) - np.min(obj.vert, axis=0))**2)) / 2

        # obj -> camera (radius)
        cam_r = r + np.random.uniform(self.cam_pos_coe_min*r, self.cam_pos_coe_max*r)
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2*np.pi)
        cam_x = center[0] + cam_r*np.sin(theta)*np.cos(phi)
        cam_y = center[1] + cam_r*np.sin(theta)*np.sin(phi)
        cam_z = center[0] + cam_r*np.cos(theta)
        self.cam_pos = [cam_x, cam_y, cam_z]

        # camera lookat
        offset_lim = cam_r*np.tan(np.radians(self.cam_angle) / 2) - r
        self.target_pos = center + np.random.uniform(-offset_lim, offset_lim, size=3)

        # light
        self.light_pos = r + np.random.uniform(self.cam_pos_coe_min*r, self.cam_pos_coe_max*r, size=3)