""" Capture 2D images of a 3D object in different point of views
"""
import moderngl
import numpy as np
from objloader import Obj
from PIL import Image, ImageOps
from pyrr import Matrix44, Vector4
from .mtlparser import MTLParser
from .material import Material

def rand_sphere(center, radius):
    theta = np.random.uniform(0, np.pi)
    phi = np.random.uniform(0, 2*np.pi)
    x = center[0] + radius*np.sin(theta)*np.cos(phi)
    y = center[1] + radius*np.sin(theta)*np.sin(phi)
    z = center[2] + radius*np.cos(theta)
    
    return np.array([x, y, z])

class Object3DCapture:
    __phong_vertex_shader = '''
        #version 330
        in vec3 inputPosition;
        in vec3 inputNormal;
        in vec2 inputTexCoord;

        uniform mat4 projection, modelview, normalMat;

        out vec3 normalInterp;
        out vec3 vertPos;
        out vec2 texCoord;

        void main(){
            gl_Position = projection * modelview * vec4(inputPosition, 1.0);
            vec4 vertPos4 = modelview * vec4(inputPosition, 1.0);
            vertPos = vec3(vertPos4) / vertPos4.w;
            normalInterp = vec3(normalMat * vec4(inputNormal, 0.0));
            texCoord = inputTexCoord;
        }
    '''
    
    __phong_fragment_shader = '''
        #version 330
        precision mediump float;

        in vec3 normalInterp;
        in vec3 vertPos;
        in vec2 texCoord;

        out vec4 f_color;

        uniform int mode;
        uniform sampler2D Texture;
        uniform vec3 lightPos;
        uniform vec3 lightColor;
        uniform float lightPower;
        uniform vec3 ambientColor;
        uniform vec3 diffuseColor;
        uniform vec3 specColor;
        uniform float shininess;

        // const vec3 lightPos = vec3(1.0,1.0,1.0);
        // const vec3 lightColor = vec3(1.0, 1.0, 1.0);
        // const float lightPower = 40.0;
        // const vec3 ambientColor = vec3(0.1, 0.0, 0.0);
        // const vec3 diffuseColor = vec3(0.5, 0.0, 0.0);
        // const vec3 specColor = vec3(1.0, 1.0, 1.0);
        // const float shininess = 16.0;
        const float screenGamma = 1.0; // Assume the monitor is calibrated to the sRGB color space

        void main() {

          vec3 normal = normalize(normalInterp);
          vec3 lightDir = lightPos - vertPos;
          float distance = length(lightDir);
          distance = distance * distance;
          lightDir = normalize(lightDir);

          float lambertian = max(dot(lightDir,normal), 0.0);
          float specular = 0.0;

          if(lambertian > 0.0) {

            vec3 viewDir = normalize(-vertPos);

            // this is blinn phong
            vec3 halfDir = normalize(lightDir + viewDir);
            float specAngle = max(dot(halfDir, normal), 0.0);
            specular = pow(specAngle, shininess);

            // this is phong (for comparison)
            if(mode == 2) {
              vec3 reflectDir = reflect(-lightDir, normal);
              specAngle = max(dot(reflectDir, viewDir), 0.0);
              // note that the exponent is different here
              specular = pow(specAngle, shininess/4.0);
            }
          }
          vec3 texColor = texture(Texture, texCoord).rgb;
          vec3 colorLinear = texColor * ambientColor +
                             texColor * diffuseColor * lambertian * lightColor * lightPower / distance +
                             specColor * specular * lightColor * lightPower / distance;
          // apply gamma correction (assume ambientColor, diffuseColor and specColor
          // have been linearized, i.e. have no gamma correction in them)
          vec3 colorGammaCorrected = pow(colorLinear, vec3(1.0/screenGamma));
          // use the gamma corrected color in the fragment
          f_color = vec4(colorGammaCorrected, 1.0);
        }
    '''
    
    """ Capture the 2D images of an 3D Object"""
    def __init__(self, wavefront_path, texture_path, material_path=None, output_size=(512, 512)):
        # context and shader program
        self.ctx = ctx = moderngl.create_standalone_context()
        ctx.enable(moderngl.DEPTH_TEST)
        prog = ctx.program(
            vertex_shader=self.__phong_vertex_shader,
            fragment_shader=self.__phong_fragment_shader
        )
        
        # load wavefront obj file and the texture
        self.obj = obj = Obj.open(wavefront_path)
        img = Image.open(texture_path).convert('RGB').transpose(Image.FLIP_TOP_BOTTOM)
        if material_path is None:
            material = Material('default')
        else:
            material = MTLParser.from_file(material_path)[0] # get only the first
        
        # mtl parser

        self.projection = prog['projection']
        self.modelview = prog['modelview']
        self.normalMat = prog['normalMat']
        self.mode = prog['mode']

        self.lightPos = prog['lightPos']
        self.lightColor = prog['lightColor']
        self.lightPower = prog['lightPower']
        self.ambientColor = prog['ambientColor']
        self.diffuseColor = prog['diffuseColor']
        self.specColor = prog['specColor']
        self.shininess = prog['shininess']


        texture = ctx.texture(img.size, 3, img.tobytes())
        texture.build_mipmaps()
        texture.use()

        vbo = ctx.buffer(obj.pack('vx vy vz nx ny nz tx ty'))
        self.vao = ctx.simple_vertex_array(prog, vbo, 'inputPosition', 'inputNormal', 'inputTexCoord')
        
        # editable parameters
        self.cam_angle = 45.0
        self.cam_pos_coe_min = 2.0
        self.cam_pos_coe_max = 10.0
        
        # setup default
        self.light_color = (1.0, 1.0, 1.0)
        self.light_power = 2.0
        self.ambien_color = material.ka
        self.diffuse_color = material.kd
        self.spec_color = material.ks
        self.shininess_value = material.ns
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
        
        cam_ratio = self.fbo.size[0] / self.fbo.size[1]
        
        p = Matrix44.perspective_projection(self.cam_angle, cam_ratio, 0.01, 1000.0)
        v = Matrix44.look_at(
            self.cam_pos,
            self.target_pos,
            self.cam_up,
        )
        t = Matrix44.from_translation(-self.obj_center)
        rx = Matrix44.from_x_rotation(self.rotate_x)
        ry = Matrix44.from_y_rotation(self.rotate_y)
        rz = Matrix44.from_z_rotation(self.rotate_z)
        m = rx*ry*rz*t

        self.projection.write(p.astype('f4').tobytes())
        mv = v*m
        self.modelview.write(mv.astype('f4').tobytes())
        n = mv.inverse.transpose()
        self.normalMat.write(n.astype('f4').tobytes())
        self.mode.value = 2 # Phong
        #lightPos.value = tuple(tuple(n*Vector4.from_vector3(light_pos))[:3])
        self.lightPos.value = tuple(self.light_pos)
        self.lightColor.value = tuple(self.light_color)
        self.lightPower.value = self.light_power
        self.ambientColor.value = tuple(self.ambien_color)
        self.diffuseColor.value = tuple(self.diffuse_color)
        self.specColor.value = tuple(self.spec_color)
        self.shininess.value = self.shininess_value
        
        self.vao.render()        
        return fbo.read()
    
    def render_image(self, clr_color=(0.0, 0.0, 0.0)):
        return Image.frombytes('RGB', self.fbo.size, self.render(clr_color), 'raw')
    
    def render_bimage(self):
        saved_light = self.light_color
        saved_ambien = self.ambien_color
        self.light_color = (0.0, 0.0, 0.0)
        self.ambien_color = (0.0, 0.0, 0.0)
        img = Image.frombytes('RGB', self.fbo.size, self.render((1.0, 1.0, 1.0)), 'raw')
        self.light_color = saved_light
        self.ambien_color = saved_ambien
        return ImageOps.invert(img.convert('L', dither=None)).convert("1")
            
    def random_context(self):
        obj = self.obj
        self.obj_center = np.mean(obj.vert, axis=0)
        r = np.sqrt(np.sum((np.max(obj.vert, axis=0) - np.min(obj.vert, axis=0))**2)) / 2

        # obj -> camera (radius)
        cam_r = r + np.random.uniform(self.cam_pos_coe_min*r, self.cam_pos_coe_max*r)
        self.cam_pos = rand_sphere((.0, .0, .0), cam_r)
        offset_lim = cam_r*np.tan(np.radians(self.cam_angle) / 2) - r
        self.target_pos = np.random.uniform(-offset_lim, offset_lim, size=3)
        self.cam_up = np.random.uniform(-1, 1, size=3)

        # light
        self.light_pos = rand_sphere((.0, .0, .0), r * self.cam_pos_coe_max)
        
        # obj rotate
        self.rotate_x, self.rotate_y, self.rotate_z = np.random.uniform(-np.pi, np.pi, size=3)