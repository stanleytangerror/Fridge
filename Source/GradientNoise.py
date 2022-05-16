import taichi as ti
from taichi.math import *
import time
import numpy as np
from Math import *

@ti.func
def _Gradient3D(h):
    h = h & 15
    d = Vec3f(0, 0, 0)

    # https://mrl.cs.nyu.edu/~perlin/paper445.pdf
    # 12 directions: cube center -> cube edge, with 4 extra di
    if h == 0 or h == 12:       d = Vec3f( 1,  1,  0)
    elif h == 1 or h == 13:     d = Vec3f(-1,  1,  0)
    elif h == 2:                d = Vec3f( 1, -1,  0)
    elif h == 3:                d = Vec3f(-1, -1,  0)
    elif h == 4:                d = Vec3f( 1,  0,  1)
    elif h == 5:                d = Vec3f(-1,  0,  1)
    elif h == 6:                d = Vec3f( 1,  0, -1)
    elif h == 7:                d = Vec3f(-1,  0, -1)
    elif h == 8:                d = Vec3f( 0,  1,  1)
    elif h == 9 or h == 14:     d = Vec3f( 0, -1,  1)
    elif h == 10:               d = Vec3f( 0,  1, -1)
    elif h == 11 or h == 15:    d = Vec3f( 0, -1, -1)

    return d

@ti.func
def _Gradient2D(h):
    h = h & 3
    d = Vec2f(0, 0)
    if h == 0:      d = Vec2f( 1,  1)
    elif h == 1:    d = Vec2f(-1,  1)
    elif h == 2:    d = Vec2f(-1, -1)
    elif h == 3:    d = Vec2f( 1, -1)
    return d

@ti.func
def _Gradient1D(h):
    h = h & 1
    return 1 if h == 0 else -1
    
@ti.data_oriented
class GradientNoise:
    # https://mrl.cs.nyu.edu/~perlin/noise/
    
    perm = [ 151,160,137,91,90,15,
           131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
           190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
           88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
           77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
           102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
           135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
           5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
           223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
           129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
           251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
           49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
           138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180   ]

    def __init__(self):
        self.HashLut = ti.field(dtype=ti.i32, shape=(len(self.perm) * 2))
        for i in range(len(self.perm)):
            self.HashLut[i] = self.perm[i]
            self.HashLut[i + len(self.perm)] = self.perm[i]

    @staticmethod
    @ti.func
    def _Fade(w): 
        return w*w*w*(w*(w*6.0-15.0)+10.0)
    
    @staticmethod
    @ti.func
    def _Lerp(t, a, b): 
        return a + t * (b - a)
    
    @staticmethod
    @ti.func
    def _Gradient3D(h):
        # directly place the function body will generate bug
        return _Gradient3D(h)
    
    @staticmethod
    @ti.func
    def _Gradient2D(h):
        # directly place the function body will generate bug
        return _Gradient2D(h)

    @staticmethod
    @ti.func
    def _Gradient1D(h):
        # directly place the function body will generate bug
        return _Gradient1D(h)
    
    @ti.func
    def Noise3D(self, x):
        # https://mrl.cs.nyu.edu/~perlin/noise/
    
        # grid
        p = ti.floor(x)
        w = x - p
        
        # quintic interpolant
        u = self._Fade(w)
        
        # HASH COORDINATES OF THE 8 CUBE CORNERS
        A = self.HashLut[p[0]  ] + p[1]
        B = self.HashLut[p[0]+1] + p[1]
        AA = self.HashLut[A  ] + p[2]
        AB = self.HashLut[A+1] + p[2]
        BA = self.HashLut[B  ] + p[2]
        BB = self.HashLut[B+1] + p[2]
    
        return  self._Lerp(u[2],self._Lerp(u[1],self._Lerp(u[0],self._Gradient3D(self.HashLut[AA  ]).dot(w + Vec3f([ 0,  0,  0]) ),  # AND ADD
                                                                self._Gradient3D(self.HashLut[BA  ]).dot(w + Vec3f([-1,  0,  0]) )), # BLENDED
                                                self._Lerp(u[0],self._Gradient3D(self.HashLut[AB  ]).dot(w + Vec3f([ 0, -1,  0]) ),  # RESULTS
                                                                self._Gradient3D(self.HashLut[BB  ]).dot(w + Vec3f([-1, -1,  0]) ))),# FROM  8
                                self._Lerp(u[1],self._Lerp(u[0],self._Gradient3D(self.HashLut[AA+1]).dot(w + Vec3f([ 0,  0, -1]) ),  # CORNERS
                                                                self._Gradient3D(self.HashLut[BA+1]).dot(w + Vec3f([-1,  0, -1]) )), # OF CUBE
                                                self._Lerp(u[0],self._Gradient3D(self.HashLut[AB+1]).dot(w + Vec3f([ 0, -1, -1]) ),
                                                                self._Gradient3D(self.HashLut[BB+1]).dot(w + Vec3f([-1, -1, -1]) ))))
    
    @ti.func
    def Noise2D(self, x):
        # https://rtouti.github.io/graphics/perlin-noise-algorithm
    
        # grid
        p = ti.floor(x)
        w = x - p
        
        # quintic interpolant
        u = self._Fade(w)
        
        # HASH COORDINATES OF THE 4 SQUARE CORNERS
        A = self.HashLut[p[0]  ] + p[1]
        B = self.HashLut[p[0]+1] + p[1]
    
        return  self._Lerp(u[1],self._Lerp(u[0],self._Gradient2D(self.HashLut[A  ]).dot(w + Vec2f([ 0,  0])) , 
                                                self._Gradient2D(self.HashLut[B  ]).dot(w + Vec2f([-1,  0])) ),
                                self._Lerp(u[0],self._Gradient2D(self.HashLut[A+1]).dot(w + Vec2f([ 0, -1])) , 
                                                self._Gradient2D(self.HashLut[B+1]).dot(w + Vec2f([-1, -1])) ))
    
    @ti.func
    def Noise1D(self, x):
        # grid
        p = ti.floor(x)
        w = x - p
        
        # quintic interpolant
        u = self._Fade(w)
        
        # HASH COORDINATES OF THE 4 SQUARE CORNERS
        A = self.HashLut[p]
    
        return  self._Lerp(u,   self._Gradient1D(self.HashLut[A  ]) * (w  ) , 
                                self._Gradient1D(self.HashLut[A+1]) * (w-1) )

if __name__ == "__main__":
    ti.init(arch=ti.vulkan)

    TexSize = 1024
    WindowSize = (TexSize, TexSize)
    windowImage = ti.Vector.field(3, float, shape=WindowSize)
    debugBuffer = ti.field(dtype=ti.f32, shape=WindowSize)
    elapsedTime = ti.field(dtype=ti.f32, shape=())
    beginTime = time.time()
    noise = GradientNoise()

    @ti.kernel
    def Generate3DNoise():
        for i, j in windowImage:
            p = Vec3f([i, j, elapsedTime[None] * 100]) * 0.01
            n = noise.Noise3D(p)
            windowImage[i, j] = Vec3f([1, 1, 1]) * (n * 0.5 + 0.5)

    @ti.kernel
    def Generate2DNoise():
        for i, j in windowImage:
            p = (Vec2f([i, j]) + elapsedTime[None] * 100) * 0.01
            n = noise.Noise2D(p)
            windowImage[i, j] = Vec3f([1, 1, 1]) * (n * 0.5 + 0.5)

    @ti.kernel
    def Generate1DNoise():
        for i, j in windowImage:
            p = (i + elapsedTime[None] * 100) * 0.01
            n = noise.Noise1D(p)
            up = 1.0 if TexSize * (n * 0.1 + 0.5) < j else 0.0
            windowImage[i, j] = Vec3f([1, 1, 1]) * up

    @ti.kernel
    def GenerateFBM():
        for i, j in windowImage:
            h = 0.0
            pos = Vec2f([i, j]) + elapsedTime[None] * 100.0
            freq = 0.001
            amp = 1.0
            for k in range(1, 20):
                m = pow(2.0, float(k))
                h += noise.Noise2D(pos * freq * m) * amp / m
            windowImage[i, j] = Vec3f([1, 1, 1]) * (h * 0.5 + 0.5)
            debugBuffer[i, j] = h

    def DebugColorBuffer():
        buf = debugBuffer.to_numpy()
        np.savetxt("temp/out.csv", buf, delimiter=',', fmt='%.2f')

    window = ti.ui.Window("Gradient Noise", WindowSize, vsync=True)

    while window.running:
        elapsedTime[None] = float(time.time() - beginTime)
        GenerateFBM()

        if window.is_pressed('p'):
            DebugColorBuffer()
            break

        window.get_canvas().set_image(windowImage)
        window.show()